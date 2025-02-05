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

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. Used to initialize the agent.
        """
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _ for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]

        self.partner_reserved_value = self.ufun.reserved_value

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
        Accept if the offer is significantly better than the reservation value.
        """
        assert self.ufun
        offer = state.current_offer
        threshold = self.ufun.reserved_value + (1.5 * (1 - state.relative_time))
        return self.ufun(offer) > threshold

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Generates a counteroffer by selecting an outcome above the opponent’s estimated reservation value.
        """
        valid_outcomes = [o for o in self.rational_outcomes if self.opponent_ufun(o) > self.partner_reserved_value]
        return random.choice(valid_outcomes) if valid_outcomes else None

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """
        Updates the estimated reservation value of the opponent based on its offers.
        """
        assert self.ufun and self.opponent_ufun
        offer = state.current_offer
        if self.opponent_ufun(offer) < self.partner_reserved_value:
            self.partner_reserved_value = float(self.opponent_ufun(offer)) / 2
        self.rational_outcomes = [o for o in self.rational_outcomes if self.opponent_ufun(o) > self.partner_reserved_value]


if __name__ == "__main__":
    from .helpers.runner import run_a_tournament

    run_a_tournament(Group14, small=True)


# python -m group14.group14
