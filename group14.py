"""
Run with nn (optional, run with rf)
"""

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from negmas.preferences import pareto_frontier
import torch
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Optional
from helpers.neural_networks_model import NeuralNetwork, TrainingPipeline
# from helpers.random_forest_model import RandomForestRegressorModel  # Import RF model class
from config.utils import load_config


class Group14(SAONegotiator):
    """
    Optimized Intelligent Negotiation Agent for ANAC 2024.
    Implements vectorized operations for faster computation.
    """

    CONFIG = load_config(file_name="group14")

    def __init__(self, *args, **kwargs):
        """
        Initializes the negotiation agent.

        Args:
            *args: Positional arguments for the base class.
            **kwargs: Keyword arguments for the base class.
        """
        super().__init__(*args, **kwargs)
        self.rational_outcomes: Optional[list] = None  # List of rational outcomes
        self.rational_outcomes_array: Optional[np.ndarray] = None  # Rational outcomes as numpy array
        self.opponent_reserved_value: float = 0.0  # Estimated opponent's reservation value
        self.state_utilities: Dict = {}  # Stores utilities computed using value iteration
        self.model: Optional[torch.nn.Module] = None  # Neural network model for opponent behavior
        # self.model: Optional[object] = None  # For the Random Forest model
        self.scaler: Optional[object] = None  # Scaler for feature transformation
        self.cached_utilities: Dict = {}  # Precomputed utilities for efficient lookup
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the trained Neural Network model and scaler with cached initialization.
        """
        try:
            if not hasattr(self, '_model_loaded'):
                input_size = 9  # Number of features used for prediction
                self.model = NeuralNetwork(input_size)
                state_dict = torch.load(TrainingPipeline.MODEL_PATH)
                self.model.load_state_dict(state_dict)
                self.model.eval()  # Set model to evaluation mode
                self.scaler = joblib.load(TrainingPipeline.SCALER_PATH)
                self._model_loaded = True  # Prevents redundant loading
        except Exception:
            # If loading fails, set model and scaler to None
            self.model = self.scaler = None

    # For Random Forest version
    # def load_model(self) -> None:
    #     """
    #     Loads the trained Random Forest model and scaler.
    #     """
    #     try:
    #         if not hasattr(self, '_model_loaded'):
    #             self.model = joblib.load(RandomForestRegressorModel.MODEL_PATH)
    #             self.scaler = joblib.load(RandomForestRegressorModel.SCALER_PATH)
    #             self._model_loaded = True  # Prevents reloading
    #     except Exception:
    #         self.model = self.scaler = None

    def on_preferences_changed(self, changes) -> None:
        """
        Handles changes in preference by recalculating rational outcomes.

        Args:
            changes: Information about the updated preferences.
        """
        if self.ufun is None:
            return  # No utility function available, exit early

        # Convert all possible outcomes into a numpy array
        outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
        utilities = np.array([self.ufun(outcome) for outcome in outcomes])
        rational_mask = utilities > self.ufun.reserved_value  # Filter rational outcomes

        # Store valid (rational) outcomes in both list and NumPy array format
        self.rational_outcomes = [outcome for outcome, is_rational in zip(outcomes, rational_mask) if is_rational]
        self.rational_outcomes_array = np.array([list(outcome) if isinstance(outcome, tuple) else outcome
                                                 for outcome in self.rational_outcomes])

        # Precompute utilities for self and opponent
        self.cached_utilities = {
            'self': np.array([self.ufun(o) for o in self.rational_outcomes]),
            'opponent': np.array([self.opponent_ufun(o) for o in self.rational_outcomes])
        }

        # Initialize opponent's reservation value and compute state utilities
        self.opponent_reserved_value = self.ufun.reserved_value
        self.state_utilities = self.value_iteration(gamma=0.9, epsilon=0.01)

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Determines the response to an offer in the negotiation.

        Args:
            state (SAOState): The current negotiation state.

        Returns:
            SAOResponse: The response decision.
        """
        offer = state.current_offer
        # Update estimated opponent's reservation value
        self.update_opponent_reserved_value(state)

        if self.ufun is None:
            # No utility function, end negotiation
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if self.acceptance_strategy(state):
            # Accept offer if conditions are met
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # Propose counteroffer
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Determines whether to accept the current offer.

        Args:
            state (SAOState): The current negotiation state.

        Returns:
            bool: True if the offer is acceptable, False otherwise.
        """
        if not self.ufun or state.current_offer is None:
            # Reject if no utility function or offer exists
            return False

        offer_utility = self.ufun(state.current_offer)
        time_factor = 1.5 * (1 - state.relative_time)  # More flexibility as time progresses
        threshold = self.ufun.reserved_value + time_factor * (1 - self.ufun.reserved_value)

        # Accept if offer meets threshold
        return offer_utility >= threshold

    def bidding_strategy(self, state: SAOState) -> Optional[Outcome]:
        """
        Determines the best counteroffer.

        Args:
            state (SAOState): The current negotiation state.

        Returns:
            Optional[Outcome]: The best counteroffer or None if no valid offer exists.
        """
        if not self.rational_outcomes:
            # No valid outcomes, return None
            return None

        opponent_utilities = self.cached_utilities['opponent']
        # Filter acceptable offers
        valid_mask = opponent_utilities > self.opponent_reserved_value

        if not np.any(valid_mask):
            # No valid counteroffers
            return None

        utilities = np.array([self.state_utilities[o] for o in self.rational_outcomes])
        valid_utilities = utilities[valid_mask]
        valid_outcomes = np.array(self.rational_outcomes)[valid_mask]

        # Select the best available outcome
        best_idx = np.argmax(valid_utilities)
        return valid_outcomes[best_idx]

    def update_opponent_reserved_value(self, state: SAOState) -> None:
        """
        Uses a trained ML model to estimate the opponent's reserved value.

        Args:
            state (SAOState): The current negotiation state.
        """
        if not all([self.ufun, self.opponent_ufun, state.current_offer, self.model, self.scaler]):
            # Skip if any essential component is missing
            return

        try:
            # Create feature vector for ML prediction
            features = np.array([[
                self.ufun(state.current_offer),
                self.ufun(state.current_offer) - self.ufun.reserved_value,
                self.opponent_ufun(state.current_offer),
                self.ufun(state.current_offer) + self.opponent_ufun(state.current_offer),
                state.relative_time,
                self.compute_nash_optimality(state.current_offer),
                self.compute_kalai_optimality(state.current_offer),
                self.compute_max_welfare_optimality(state.current_offer),
                float(self.is_pareto_optimal(state.current_offer))
            ]])

            # Transform features and predict reserved value
            features_df = pd.DataFrame(features, columns=self.CONFIG["ml_features"])
            features_scaled = self.scaler.transform(features_df)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

            with torch.no_grad():
                predicted_value = self.model(features_tensor).item()
                self.opponent_reserved_value = (self.opponent_reserved_value + predicted_value) / 2  # Smoothing update

        except Exception:
            pass

    # For the Random Forest version
    # def update_opponent_reserved_value(self, state: SAOState) -> None:
    #     """
    #     Updates the partner's reserved value using the Random Forest Regressor.
    #
    #     Args:
    #         state (SAOState): The current negotiation state.
    #     """
    #     if not all([self.ufun, self.opponent_ufun, state.current_offer, self.model, self.scaler]):
    #         return
    #
    #     try:
    #         # Construct feature array
    #         features = np.array([[
    #             self.ufun(state.current_offer),  # Utility
    #             self.ufun(state.current_offer) - self.ufun.reserved_value,  # Advantage
    #             self.opponent_ufun(state.current_offer),  # Partner Welfare
    #             self.ufun(state.current_offer) + self.opponent_ufun(state.current_offer),  # Welfare
    #             state.relative_time,  # Time
    #             self.compute_nash_optimality(state.current_offer),  # Nash Optimality
    #             self.compute_kalai_optimality(state.current_offer),  # Kalai Optimality
    #             self.compute_max_welfare_optimality(state.current_offer),  # Max Welfare Optimality
    #             float(self.is_pareto_optimal(state.current_offer))  # Pareto Optimality
    #         ]])
    #
    #         # Convert features array to DataFrame with proper column names
    #         features_df = pd.DataFrame(features, columns=self.CONFIG["ml_features"])
    #
    #         # Transform features using the scaler
    #         features_scaled = self.scaler.transform(features_df)
    #
    #         # Predict using Random Forest
    #         predicted_value = self.model.predict(features_scaled)[0]
    #
    #         # Smoothing update
    #         self.opponent_reserved_value = (self.opponent_reserved_value + predicted_value) / 2
    #
    #     except Exception:
    #         pass

    def value_iteration(self, gamma: float = 0.9, epsilon: float = 0.01) -> Dict:
        """
        Uses value iteration to determine optimal long-term utilities.

        This function helps the agent evaluate each outcome not just based on its immediate
        reward (utility) but also by considering future potential gains. It iteratively
        updates utility values until they converge to a stable state.

        Args:
            gamma (float): Discount factor (controls how much future rewards matter).
            epsilon (float): Convergence threshold (stopping condition).

        Returns:
            Dict: Mapping of outcomes to their computed utilities.
        """
        # Retrieve precomputed utility values for all rational outcomes
        utilities = self.cached_utilities['self']

        # Initialize utility estimates for all outcomes to 0
        U = np.zeros(len(self.rational_outcomes))
        U_prime = U.copy()  # Copy to store updated values
        delta = float('inf')  # Initialize a high difference value to start the loop

        # Value iteration loop: continues until utility estimates converge
        while delta >= epsilon * (1 - gamma) / gamma:
            # Store the previous iteration's values
            U = U_prime.copy()
            # Immediate reward is the utility of the outcome
            reward = utilities
            # Bellman equation update: combines immediate rewards and future value
            U_prime = reward + gamma * np.max(U)
            # Compute the difference between iterations (for convergence check)
            delta = np.max(np.abs(U_prime - U))

        # Return a dictionary mapping each outcome to its computed long-term utility
        return dict(zip(self.rational_outcomes, U_prime))

    def compute_nash_optimality(self, offer: Outcome) -> float:
        """
        Computes the Nash optimality of a given offer.

        Nash optimality is calculated as the product of the utility values of both negotiators.
        Higher values indicate better trade-offs for both parties.

        Args:
            offer (Outcome): The offer for which Nash optimality is computed.

        Returns:
            float: The Nash optimality score.
        """
        return self.ufun(offer) * self.opponent_ufun(offer)

    def compute_kalai_optimality(self, offer: Outcome) -> float:
        """
        Computes the Kalai-Smorodinsky optimality of a given offer.

        Kalai-Smorodinsky optimality ensures fairness by maintaining proportionality
        relative to each negotiator’s maximum possible utility.

        Args:
            offer (Outcome): The offer for which Kalai-Smorodinsky optimality is computed.

        Returns:
            float: The Kalai-Smorodinsky optimality score (range: 0 to 1).
        """
        if not self.cached_utilities:
            return 0.0  # Default value if utilities are not cached

        u_max = np.max(self.cached_utilities['self'])  # Maximum possible utility for self
        v_max = np.max(self.cached_utilities['opponent'])  # Maximum possible utility for opponent

        return min(self.ufun(offer) / u_max, self.opponent_ufun(offer) / v_max)

    def compute_max_welfare_optimality(self, offer: Outcome) -> float:
        """
        Computes the welfare optimality of a given offer.

        Welfare optimality is defined as the ratio of the total utility of the offer
        to the maximum possible total utility of any offer.

        Args:
            offer (Outcome): The offer for which welfare optimality is computed.

        Returns:
            float: The welfare optimality score (range: 0 to 1).
        """
        if not self.cached_utilities:
            return 0.0  # Default value if utilities are not cached

        total_welfare = self.cached_utilities['self'] + self.cached_utilities['opponent']
        max_welfare = np.max(total_welfare)  # Maximum possible combined utility
        current_welfare = self.ufun(offer) + self.opponent_ufun(offer)

        return current_welfare / max_welfare if max_welfare > 0 else 0.0

    def is_pareto_optimal(self, offer: Outcome) -> bool:
        """
        Determines whether a given offer is Pareto optimal.

        An offer is Pareto optimal if there is no other offer that would increase
        one player's utility without decreasing the other's.

        Args:
            offer (Outcome): The offer to check for Pareto optimality.

        Returns:
            bool: True if the offer is Pareto optimal, False otherwise.
        """
        pareto_outcomes = pareto_frontier(self.ufun, self.opponent_ufun)
        return offer in pareto_outcomes


if __name__ == "__main__":
    from helpers.runner import run_a_tournament
    run_a_tournament(Group14, small=True)

#  python -m group14.group14
