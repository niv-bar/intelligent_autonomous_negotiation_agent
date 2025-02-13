import pathlib
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import Optional

from config.utils import load_config


class RandomForestRegressorModel:
    """
    Training pipeline for the Group14 Negotiation Agent using a Random Forest Regressor.

    Handles data loading, preprocessing, model training, and saving of trained components.
    """

    # File paths for saving the model and scaler
    MODEL_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "models" / "group14_rf_model.pkl"
    SCALER_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "models" / "group14_rf_scaler.pkl"
    TRAINING_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "training_data"

    # Load configuration for features and target variable
    CONFIG = load_config(file_name="group14")

    def __init__(self, max_depth: int = 10, n_estimators: int = 100, min_samples_split: int = 5):
        """
        Initializes the training pipeline.

        Args:
            max_depth (int, optional): Maximum depth of the decision trees. Default is 10.
            n_estimators (int, optional): Number of trees in the forest. Default is 100.
            min_samples_split (int, optional): Minimum samples required to split a node. Default is 5.
        """
        self.scaler = StandardScaler()
        self.model: Optional[RandomForestRegressor] = None
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split

    def build_model(self) -> None:
        """
        Builds and initializes the Random Forest model.
        """
        self.model = RandomForestRegressor(
            random_state=42,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            n_jobs=-1  # Use all available CPU cores
        )

    def train_model(self, test_size: float = 0.2) -> None:
        """
        Trains the Random Forest regression model using the provided dataset.

        Args:
            test_size (float, optional): Proportion of data to be used for testing. Default is 0.2 (20%).
        """
        # Load and preprocess dataset
        all_scores_combined = pd.concat([
            pd.read_csv(self.TRAINING_PATH / f"all_scores_v{i}.csv")
            for i in range(1, 6)
        ], ignore_index=True)

        df = all_scores_combined[all_scores_combined['strategy'] != 'group14'].copy()

        # Drop rows with missing values
        df = df.dropna(subset=[self.CONFIG["target_feature"]] + self.CONFIG["ml_features"])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[self.CONFIG["ml_features"]], df[self.CONFIG["target_feature"]],
            test_size=test_size,
            random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize and train the model
        self.build_model()
        self.model.fit(X_train_scaled, y_train)

        # Save the trained model and scaler
        joblib.dump(self.model, self.MODEL_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)

        print("Training completed. Model and scaler saved successfully.")


if __name__ == "__main__":
    RandomForestRegressorModel().train_model()


""" 
Copy it to group14 to run with rf regressor 
"""

# from negmas.outcomes import Outcome
# from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
# import pandas as pd
# import joblib
# import numpy as np
# from typing import Dict, Optional
# from .helpers.random_forest_regressor import RandomForestRegressorModel  # Import RF model class
# from config.utils import load_config
#
#
# class Group14(SAONegotiator):
#     """
#     Optimized Intelligent Negotiation Agent for ANAC 2024.
#     Implements vectorized operations for faster computation using a Random Forest Regressor.
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
#         self.model: Optional[object] = None  # Random Forest model
#         self.scaler: Optional[object] = None  # Scaler for feature transformation
#         self.cached_utilities: Dict = {}  # Precomputed utilities for efficient lookup
#         self.load_model()
#
#     def load_model(self) -> None:
#         """
#         Loads the trained Random Forest model and scaler.
#         """
#         try:
#             if not hasattr(self, '_model_loaded'):
#                 self.model = joblib.load(RandomForestRegressorModel.MODEL_PATH)
#                 self.scaler = joblib.load(RandomForestRegressorModel.SCALER_PATH)
#                 self._model_loaded = True  # Prevents reloading
#         except Exception as e:
#             print(f"Error loading Random Forest model: {e}")
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
#         Updates the partner's reserved value using the Random Forest Regressor.
#
#         Args:
#             state (SAOState): The current negotiation state.
#         """
#         if not all([self.ufun, self.opponent_ufun, state.current_offer, self.model, self.scaler]):
#             return
#
#         try:
#             # Construct feature array
#             features = np.array([[
#                 self.ufun(state.current_offer),  # Utility
#                 self.ufun(state.current_offer) - self.ufun.reserved_value,  # Advantage
#                 self.opponent_ufun(state.current_offer),  # Partner Welfare
#                 self.ufun(state.current_offer) + self.opponent_ufun(state.current_offer),  # Welfare
#                 state.relative_time,  # Time
#                 self.compute_nash_optimality(state.current_offer),  # Nash Optimality
#                 self.compute_kalai_optimality(state.current_offer),  # Kalai Optimality
#                 self.compute_max_welfare_optimality(state.current_offer),  # Max Welfare Optimality
#                 float(self.is_pareto_optimal(state.current_offer))  # Pareto Optimality
#             ]])
#
#             # Convert features array to DataFrame with proper column names
#             features_df = pd.DataFrame(features, columns=self.CONFIG["ml_features"])
#
#             # Transform features using the scaler
#             features_scaled = self.scaler.transform(features_df)
#
#             # Predict using Random Forest
#             predicted_value = self.model.predict(features_scaled)[0]
#
#             # Smoothing update
#             self.partner_reserved_value = (self.partner_reserved_value + predicted_value) / 2
#
#         except Exception as e:
#             print(f"Error in update_partner_reserved_value: {e}")
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

