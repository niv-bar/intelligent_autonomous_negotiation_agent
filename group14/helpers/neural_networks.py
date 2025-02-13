import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pathlib
from typing import Optional

from config.utils import load_config


class NeuralNetwork(nn.Module):
    """
    Deep Learning model for regression-based negotiation strategy prediction.

    This neural network consists of three hidden layers with batch normalization,
    LeakyReLU activations, and dropout for regularization.
    """

    def __init__(self, input_size: int, dropout_rate: float = 0.2):
        """
        Initializes the neural network architecture.

        Args:
            input_size (int): Number of input features.
            dropout_rate (float, optional): Dropout probability for regularization. Default is 0.2.
        """
        super(NeuralNetwork, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_size)

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.output_layer = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Predicted output value.
        """
        x = self.batch_norm1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output_layer(x)


class TrainingPipeline:
    """
    Training pipeline for the Group14 Negotiation Agent using a deep learning model.

    Handles data loading, preprocessing, model training, and saving of trained components.
    """

    # File paths for saving the model and scaler
    MODEL_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "models" / "group14_nn_model.pth"
    SCALER_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "models" / "group14_nn_scaler.pkl"
    TRAINING_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "training_data"

    # Load configuration for features and target variable
    CONFIG = load_config(file_name="group14")

    def __init__(self, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Initializes the training pipeline.

        Args:
            batch_size (int, optional): Number of samples per batch during training. Default is 32.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.model: Optional[NeuralNetwork] = None
        self.criterion = nn.HuberLoss(delta=1.0)  # Huber loss is more robust to outliers

    def build_model(self, input_size: int) -> None:
        """
        Builds and initializes the deep learning model.

        Args:
            input_size (int): Number of input features.
        """
        self.model = NeuralNetwork(input_size)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01  # L2 regularization
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    def train_model(self, epochs: int = 100, early_stopping_patience: int = 10) -> None:
        """
        Trains the model using the provided dataset.

        Implements early stopping and learning rate scheduling to optimize training.

        Args:
            epochs (int, optional): Number of training epochs. Default is 100.
            early_stopping_patience (int, optional): Number of epochs to wait before early stopping. Default is 10.
        """
        # Load and preprocess dataset
        all_scores_combined = pd.concat([
            pd.read_csv(f"{self.TRAINING_PATH}/all_scores_v{i}.csv")
            for i in range(1, 6)
        ], ignore_index=True)

        df = all_scores_combined[all_scores_combined['strategy'] != 'group14'].copy()

        # Drop rows with missing values
        df = df.dropna(subset=[self.CONFIG["target_feature"]] + self.CONFIG["ml_features"])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[self.CONFIG["ml_features"]], df[self.CONFIG["target_feature"]],
            test_size=0.2,
            random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train.values).reshape(-1, 1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test.values).reshape(-1, 1)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Initialize model
        self.build_model(len(self.CONFIG["ml_features"]))

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = self.model(batch_X)
                    val_loss += self.criterion(outputs, batch_y).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)

            # Learning rate adjustment
            self.scheduler.step(avg_val_loss)

            # Early stopping mechanism
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.MODEL_PATH)  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save scaler for later inference
        joblib.dump(self.scaler, self.SCALER_PATH)
        print("Training completed. Model and scaler saved successfully.")


if __name__ == "__main__":
    TrainingPipeline().train_model()

""" 
Copy it to group14 to run with nn 
"""

# from negmas.outcomes import Outcome
# from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
# import torch
# import pandas as pd
# import joblib
# import numpy as np
# from typing import Dict, Optional
# from .helpers.neural_networks import NeuralNetwork, TrainingPipeline
# from .helpers.random_forest_regressor import RandomForestRegressorModel  # Import RF model class
# from config.utils import load_config
#
#
# class Group14(SAONegotiator):
#     """
#     Optimized Intelligent Negotiation Agent for ANAC 2024.
#     Implements vectorized operations for faster computation.
#     """
#
#     CONFIG = load_config(file_name="group14")
#
#     def __init__(self, *args, **kwargs):
#         """
#         Initializes the negotiation agent.
#
#         Args:
#             *args: Positional arguments for the base class.
#             **kwargs: Keyword arguments for the base class.
#         """
#         super().__init__(*args, **kwargs)
#         self.rational_outcomes: Optional[list] = None  # List of rational outcomes
#         self.rational_outcomes_array: Optional[np.ndarray] = None  # Rational outcomes as numpy array
#         self.partner_reserved_value: float = 0.0  # Estimated opponent's reservation value
#         self.state_utilities: Dict = {}  # Stores utilities computed using value iteration
#         self.model: Optional[torch.nn.Module] = None  # Neural network model for opponent behavior
#         self.scaler: Optional[object] = None  # Scaler for feature transformation
#         self.cached_utilities: Dict = {}  # Precomputed utilities for efficient lookup
#         self.load_model()
#
#     def load_model(self) -> None:
#         """
#         Loads the trained model and scaler with cached initialization.
#         """
#         try:
#             if not hasattr(self, '_model_loaded'):
#                 input_size = 9  # Number of features used for prediction
#                 self.model = NeuralNetwork(input_size)
#                 state_dict = torch.load(TrainingPipeline.MODEL_PATH)
#                 self.model.load_state_dict(state_dict)
#                 self.model.eval()
#                 self.scaler = joblib.load(TrainingPipeline.SCALER_PATH)
#                 self._model_loaded = True  # Prevents reloading
#         except Exception:
#             self.model = self.scaler = None
#
#     def on_preferences_changed(self, changes) -> None:
#         """
#         Handles changes in preference by recalculating rational outcomes.
#
#         Args:
#             changes: Information about the updated preferences.
#         """
#         if self.ufun is None:
#             return
#
#         # Convert all possible outcomes into a numpy array
#         outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
#         utilities = np.array([self.ufun(outcome) for outcome in outcomes])
#         rational_mask = utilities > self.ufun.reserved_value
#
#         # Filter rational outcomes
#         self.rational_outcomes = [outcome for outcome, is_rational in zip(outcomes, rational_mask) if is_rational]
#         self.rational_outcomes_array = np.array([list(outcome) if isinstance(outcome, tuple) else outcome
#                                                  for outcome in self.rational_outcomes])
#
#         # Precompute utilities for self and opponent
#         self.cached_utilities = {
#             'self': np.array([self.ufun(o) for o in self.rational_outcomes]),
#             'opponent': np.array([self.opponent_ufun(o) for o in self.rational_outcomes])
#         }
#
#         self.partner_reserved_value = self.ufun.reserved_value
#         self.state_utilities = self.value_iteration(gamma=0.9, epsilon=0.01)
#
#     def __call__(self, state: SAOState) -> SAOResponse:
#         """
#         Determines the response to an offer in the negotiation.
#
#         Args:
#             state (SAOState): The current negotiation state.
#
#         Returns:
#             SAOResponse: The response decision.
#         """
#         offer = state.current_offer
#         self.update_partner_reserved_value(state)
#
#         if self.ufun is None:
#             return SAOResponse(ResponseType.END_NEGOTIATION, None)
#
#         if self.acceptance_strategy(state):
#             return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
#
#         return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))
#
#     def acceptance_strategy(self, state: SAOState) -> bool:
#         """
#         Determines whether to accept the current offer.
#
#         Args:
#             state (SAOState): The current negotiation state.
#
#         Returns:
#             bool: True if the offer is acceptable, False otherwise.
#         """
#         if not self.ufun or state.current_offer is None:
#             return False
#
#         offer_utility = self.ufun(state.current_offer)
#         time_factor = 1.5 * (1 - state.relative_time)
#         threshold = self.ufun.reserved_value + time_factor * (1 - self.ufun.reserved_value)
#         return offer_utility >= threshold
#
#     def bidding_strategy(self, state: SAOState) -> Optional[Outcome]:
#         """
#         Determines the best counteroffer.
#
#         Args:
#             state (SAOState): The current negotiation state.
#
#         Returns:
#             Optional[Outcome]: The best counteroffer or None if no valid offer exists.
#         """
#         if not self.rational_outcomes:
#             return None
#
#         opponent_utilities = self.cached_utilities['opponent']
#         valid_mask = opponent_utilities > self.partner_reserved_value
#
#         if not np.any(valid_mask):
#             return None
#
#         utilities = np.array([self.state_utilities[o] for o in self.rational_outcomes])
#         valid_utilities = utilities[valid_mask]
#         valid_outcomes = np.array(self.rational_outcomes)[valid_mask]
#
#         best_idx = np.argmax(valid_utilities)
#         return valid_outcomes[best_idx]
#
#     def update_partner_reserved_value(self, state: SAOState) -> None:
#         """
#         Updates the partner's reserved value using machine learning.
#
#         Args:
#             state (SAOState): The current negotiation state.
#         """
#         if not all([self.ufun, self.opponent_ufun, state.current_offer, self.model, self.scaler]):
#             return
#
#         try:
#             features = np.array([[
#                 self.ufun(state.current_offer),
#                 self.ufun(state.current_offer) - self.ufun.reserved_value,
#                 self.opponent_ufun(state.current_offer),
#                 self.ufun(state.current_offer) + self.opponent_ufun(state.current_offer),
#                 state.relative_time,
#                 self.compute_nash_optimality(state.current_offer),
#                 self.compute_kalai_optimality(state.current_offer),
#                 self.compute_max_welfare_optimality(state.current_offer),
#                 float(self.is_pareto_optimal(state.current_offer))
#             ]])
#
#             features_df = pd.DataFrame(features, columns=self.CONFIG["ml_features"])
#             features_scaled = self.scaler.transform(features_df)
#
#             features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
#
#             with torch.no_grad():
#                 predicted_value = self.model(features_tensor).item()
#                 self.partner_reserved_value = (self.partner_reserved_value + predicted_value) / 2  # Smoothing update
#
#         except Exception:
#             pass
#
#     def value_iteration(self, gamma: float = 0.9, epsilon: float = 0.01) -> Dict:
#         """
#         Performs value iteration for decision optimization.
#
#         Args:
#             gamma (float): Discount factor.
#             epsilon (float): Convergence threshold.
#
#         Returns:
#             Dict: Mapping of outcomes to utility values.
#         """
#         utilities = self.cached_utilities['self']
#         U = np.zeros(len(self.rational_outcomes))
#         U_prime = U.copy()
#         delta = float('inf')
#
#         while delta >= epsilon * (1 - gamma) / gamma:
#             U = U_prime.copy()
#             reward = utilities
#             U_prime = reward + gamma * np.max(U)
#             delta = np.max(np.abs(U_prime - U))
#
#         return dict(zip(self.rational_outcomes, U_prime))
#
#     def compute_nash_optimality(self, offer: Outcome) -> float:
#         """
#         Computes the Nash optimality of a given offer.
#
#         Nash optimality is calculated as the product of the utility values of both negotiators.
#         Higher values indicate better trade-offs for both parties.
#
#         Args:
#             offer (Outcome): The offer for which Nash optimality is computed.
#
#         Returns:
#             float: The Nash optimality score.
#         """
#         return self.ufun(offer) * self.opponent_ufun(offer)
#
#     def compute_kalai_optimality(self, offer: Outcome) -> float:
#         """
#         Computes the Kalai-Smorodinsky optimality of a given offer.
#
#         Kalai-Smorodinsky optimality ensures fairness by maintaining proportionality
#         relative to each negotiator’s maximum possible utility.
#
#         Args:
#             offer (Outcome): The offer for which Kalai-Smorodinsky optimality is computed.
#
#         Returns:
#             float: The Kalai-Smorodinsky optimality score (range: 0 to 1).
#         """
#         if not self.cached_utilities:
#             return 0.0  # Default value if utilities are not cached
#
#         u_max = np.max(self.cached_utilities['self'])  # Maximum possible utility for self
#         v_max = np.max(self.cached_utilities['opponent'])  # Maximum possible utility for opponent
#
#         return min(self.ufun(offer) / u_max, self.opponent_ufun(offer) / v_max)
#
#     def compute_max_welfare_optimality(self, offer: Outcome) -> float:
#         """
#         Computes the welfare optimality of a given offer.
#
#         Welfare optimality is defined as the ratio of the total utility of the offer
#         to the maximum possible total utility of any offer.
#
#         Args:
#             offer (Outcome): The offer for which welfare optimality is computed.
#
#         Returns:
#             float: The welfare optimality score (range: 0 to 1).
#         """
#         if not self.cached_utilities:
#             return 0.0  # Default value if utilities are not cached
#
#         total_welfare = self.cached_utilities['self'] + self.cached_utilities['opponent']
#         max_welfare = np.max(total_welfare)  # Maximum possible combined utility
#         current_welfare = self.ufun(offer) + self.opponent_ufun(offer)
#
#         return current_welfare / max_welfare if max_welfare > 0 else 0.0
#
#     def is_pareto_optimal(self, offer: Outcome) -> bool:
#         """
#         Determines whether a given offer is Pareto optimal.
#
#         An offer is Pareto optimal if there is no other offer that would increase
#         one player's utility without decreasing the other's.
#
#         Args:
#             offer (Outcome): The offer to check for Pareto optimality.
#
#         Returns:
#             bool: True if the offer is Pareto optimal, False otherwise.
#         """
#         if not self.cached_utilities:
#             return False  # Default to False if utilities are not cached
#
#         offer_utility_self = self.ufun(offer)  # Utility of the offer for self
#         offer_utility_opponent = self.opponent_ufun(offer)  # Utility of the offer for opponent
#
#         utilities_self = self.cached_utilities['self']  # Array of self utilities
#         utilities_opponent = self.cached_utilities['opponent']  # Array of opponent utilities
#
#         # Check if any offer dominates the given offer
#         dominated = np.logical_and(
#             utilities_self >= offer_utility_self,
#             utilities_opponent > offer_utility_opponent
#         ) | np.logical_and(
#             utilities_self > offer_utility_self,
#             utilities_opponent >= offer_utility_opponent
#         )
#
#         return not np.any(dominated)  # True if no other offer dominates the given one
#
#
# if __name__ == "__main__":
#     from .helpers.runner import run_a_tournament
#     run_a_tournament(Group14, small=True)
#     #  python -m group14.group14
