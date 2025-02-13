from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class Group14(SAONegotiator):
    """
    Intelligent Negotiation Agent for ANAC 2024.
    Implements offering and acceptance strategies while adapting to opponent behavior.
    """

    rational_outcomes = tuple()
    partner_reserved_value = 0
    state_utilities = {}
    regressor = None
    scaler = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_regression_model()

    def train_regression_model(self):
        """
        Trains a regression model to predict the opponent's reserved value.
        """
        all_scores = pd.read_csv("all_scores.csv")

        df = all_scores.copy()
        features = ['utility', 'advantage', 'partner_welfare', 'welfare', 'time',
                    'nash_optimality', 'kalai_optimality', 'max_welfare_optimality', 'pareto_optimality']
        target = 'reserved_value'
        df = df.dropna(subset=[target] + features)
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.regressor = LinearRegression()
        self.regressor.fit(X_train_scaled, y_train)

        y_pred = self.regressor.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Regression Model - MSE: {mse}, R² Score: {r2}")

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. Used to initialize the agent.
        """
        if self.ufun is None:
            return

        self.rational_outcomes = [
            outcome for outcome in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(outcome) > self.ufun.reserved_value
        ]

        self.partner_reserved_value = self.ufun.reserved_value
        self.state_utilities = self.value_iteration(gamma=0.9, epsilon=0.01)

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Main negotiation loop. Determines whether to accept, reject, or propose a counteroffer.
        """
        offer = state.current_offer
        self.update_partner_reserved_value(state)

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Determines whether to accept an offer.
        Accepts if the offer exceeds a dynamic threshold based on negotiation progress.
        """
        assert self.ufun
        offer = state.current_offer
        time_factor = 1.5 * (1 - state.relative_time)
        threshold = self.ufun.reserved_value + time_factor * (1 - self.ufun.reserved_value)
        return self.ufun(offer) >= threshold

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Generates a counteroffer based on Value Iteration computed utilities.
        """
        valid_outcomes = [o for o in self.rational_outcomes if self.opponent_ufun(o) > self.partner_reserved_value]

        if not valid_outcomes:
            return None

        # Select the best outcome based on Value Iteration computed utilities
        best_outcome = max(valid_outcomes, key=lambda o: self.state_utilities.get(o, 0))
        return best_outcome

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """
        Updates the estimated reservation value of the opponent based on its offers.
        """
        assert self.ufun and self.opponent_ufun
        offer = state.current_offer
        if offer is None:
            return


        # print("utility", self.ufun(state.current_offer))
        # print("reserved_value", self.ufun.reserved_value)
        # print("advantage", (self.ufun(state.current_offer) - self.ufun.reserved_value))
        # print("partner_welfare", self.opponent_ufun(state.current_offer))
        # print("welfare", (self.ufun(state.current_offer) + self.opponent_ufun(state.current_offer)))
        # print("time", state.relative_time)
        # print("nash_optimality", self.compute_nash_optimality(state.current_offer))
        # print("kalai_optimality", self.compute_kalai_optimality(state.current_offer))
        # print("max_welfare_optimality", self.compute_max_welfare_optimality(state.current_offer))
        # print("pareto_optimality", self.is_pareto_optimal(state.current_offer))

            # Create a DataFrame with the same feature names
            opponent_features = pd.DataFrame([[
                self.ufun(state.current_offer),  # utility
                (self.ufun(state.current_offer) - self.ufun.reserved_value),  # advantage
                self.opponent_ufun(state.current_offer),  # partner_welfare
                (self.ufun(state.current_offer) + self.opponent_ufun(state.current_offer)),  # welfare
                state.relative_time,  # time
                self.compute_nash_optimality(state.current_offer),  # nash_optimality
                self.compute_kalai_optimality(state.current_offer),  # kalai_optimality
                self.compute_max_welfare_optimality(state.current_offer),  # max_welfare_optimality
                self.is_pareto_optimal(state.current_offer)  # pareto_optimality
            ]], columns=[
                'utility', 'advantage', 'partner_welfare', 'welfare', 'time',
                'nash_optimality', 'kalai_optimality', 'max_welfare_optimality', 'pareto_optimality'
            ])

            # Standardize input features using the same scaler
            opponent_features_scaled = self.scaler.transform(opponent_features)

            # Predict reserved value
            predicted_value = self.regressor.predict(opponent_features_scaled)[0]

            # Update estimated opponent's reservation value
            self.partner_reserved_value = (self.partner_reserved_value + predicted_value) / 2

    def value_iteration(self, gamma=0.9, epsilon=0.01):
        """
        Value Iteration - https://people.engr.tamu.edu/guni/csce421/files/AI_Russell_Norvig.pdf
        page 652
        Implements the Value Iteration algorithm to estimate the best possible outcomes.
        """
        U = {outcome: 0 for outcome in self.rational_outcomes}
        U_prime = U.copy()
        delta = float('inf')

        while delta >= epsilon * (1 - gamma) / gamma:
            U = U_prime.copy()
            delta = 0

            for outcome in self.rational_outcomes:
                reward = self.ufun(outcome)
                U_prime[outcome] = reward + gamma * max(U.values(), default=0)
                delta = max(delta, abs(U_prime[outcome] - U[outcome]))

        return U_prime

    def compute_nash_optimality(self, offer: Outcome) -> float:
        """
        Computes how close the current offer is to the Nash Bargaining solution.
        """
        if not self.ufun or not self.opponent_ufun:
            return None
        return self.ufun(offer) * self.opponent_ufun(offer)  # Nash product

    def compute_kalai_optimality(self, offer: Outcome) -> float:
        """
        Approximates Kalai-Smorodinsky optimality.
        """
        if not self.ufun or not self.opponent_ufun:
            return None
        u_max = max(self.ufun(o) for o in self.rational_outcomes)
        v_max = max(self.opponent_ufun(o) for o in self.rational_outcomes)
        return min(self.ufun(offer) / u_max, self.opponent_ufun(offer) / v_max)

    def compute_max_welfare_optimality(self, offer: Outcome) -> float:
        """
        Checks if the offer maximizes the total welfare.
        """
        if not self.ufun or not self.opponent_ufun:
            return None
        max_welfare = max(self.ufun(o) + self.opponent_ufun(o) for o in self.rational_outcomes)
        return (self.ufun(offer) + self.opponent_ufun(offer)) / max_welfare

    def is_pareto_optimal(self, offer: Outcome) -> bool:
        """
        Determines if the offer is Pareto optimal.
        """
        if not self.ufun or not self.opponent_ufun:
            return False
        for o in self.rational_outcomes:
            if (self.ufun(o) >= self.ufun(offer) and self.opponent_ufun(o) > self.opponent_ufun(offer)) or \
                    (self.ufun(o) > self.ufun(offer) and self.opponent_ufun(o) >= self.opponent_ufun(offer)):
                return False  # There exists a better offer for at least one party
        return True


if __name__ == "__main__":
    from .helpers.runner import run_a_tournament
    run_a_tournament(Group14, small=True)
    #  python -m group14.group14
