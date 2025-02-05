import random

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


class Group14(SAONegotiator):
    """
    Intelligent Negotiation Agent for ANAC 2024.
    Implements offering and acceptance strategies while adapting to opponent behavior.
    """

    rational_outcomes = tuple()
    partner_reserved_value = 0
    state_utilities = {}

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

        opponent_offer_value = self.opponent_ufun(offer)
        if opponent_offer_value < self.partner_reserved_value:
            self.partner_reserved_value = (self.partner_reserved_value + opponent_offer_value) / 2

        self.rational_outcomes = [o for o in self.rational_outcomes if self.opponent_ufun(o) > self.partner_reserved_value]

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


if __name__ == "__main__":
    from .helpers.runner import run_a_tournament
    run_a_tournament(Group14, small=True)
