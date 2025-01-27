class Group14:
    """
    A negotiation agent designed for bilateral negotiations using the NegMAS platform.
    """

    def __init__(self):
        self.agent_name = "Group14"
        self.opponent_model = None
        self.initial_utility_threshold = 0.7
        self.final_utility_threshold = 0.4
        self.total_utility = 0
        self.num_offers_accepted = 0

    def initialize(self):
        """
        Method to initialize the agent before starting a negotiation session.
        """
        print(f"{self.agent_name} is initialized.")

    def propose(self, current_offer, time_remaining=1.0):
        """
        Generate a counter-offer during negotiations.
        :param current_offer: The offer made by the opponent.
        :param time_remaining: A float between 0 and 1 representing the remaining time.
        :return: A counter-offer.
        """
        self.update_opponent_model(current_offer)

        # Predict opponent preferences
        predicted_preferences = self.predict_opponent_preferences()

        # Start with an initial offer if no current offer
        if not current_offer:
            return {"price": 100, "quality": "high"}

        # Generate counter-offer with time-based concessions
        new_offer = current_offer.copy()
        for key, value in predicted_preferences.items():
            if key == "price":
                # Concede more aggressively as time_remaining decreases
                initial_price = 100  # Example: Starting price
                minimum_price = 50  # Example: Minimum acceptable price
                concession_rate = (initial_price - minimum_price) * (1 - time_remaining)
                new_offer[key] = max(minimum_price, current_offer[key] - concession_rate)
            else:
                # Align with opponent's preferences on non-price attributes
                new_offer[key] = value
        return new_offer

    def accept(self, current_offer, time_remaining=1.0):
        """
        Decide whether to accept an offer.
        :param current_offer: The offer made by the opponent.
        :param time_remaining: A float between 0 and 1 representing the remaining time.
        :return: True if the offer is acceptable, False otherwise.
        """
        if not current_offer:
            return False

        # Predict opponent preferences
        predicted_preferences = self.predict_opponent_preferences()

        # Calculate utility of the offer based on predicted preferences
        utility = 0
        for key, value in current_offer.items():
            if predicted_preferences.get(key) == value:
                utility += 0.5  # Assign a weight for matching preferences

            # Calculate dynamic threshold
            threshold = self.initial_utility_threshold * time_remaining + self.final_utility_threshold * (1 - time_remaining)

        accepted = utility >= threshold
        if accepted:
            self.total_utility += utility
            self.num_offers_accepted += 1
        return accepted

    def get_metrics(self):
        """
        Retrieve performance metrics for the agent.
        """
        avg_utility = self.total_utility / max(1, self.num_offers_accepted)
        return {"total_utility": self.total_utility, "average_utility": avg_utility}

    def update_opponent_model(self, offer):
        """
        Update the opponent's model based on their offers.
        :param offer: The opponent's offer.
        """
        if not self.opponent_model:
            self.opponent_model = {"offer_history": [], "frequency": {}}

        # Record the offer
        self.opponent_model["offer_history"].append(offer)

        # Update frequency counts for each key-value pair in the offer
        for key, value in offer.items():
            if key not in self.opponent_model["frequency"]:
                self.opponent_model["frequency"][key] = {}
            if value not in self.opponent_model["frequency"][key]:
                self.opponent_model["frequency"][key][value] = 0
            self.opponent_model["frequency"][key][value] += 1

    def predict_opponent_preferences(self):
        """
        Predict the opponent's preferences based on their offer history.
        :return: A dictionary of predicted preferences.
        """
        if not self.opponent_model:
            return {}

        # Estimate preferences based on frequency
        preferences = {}
        for key, value_counts in self.opponent_model["frequency"].items():
            preferences[key] = max(value_counts, key=value_counts.get)  # Most frequent value
        return preferences


if __name__ == "__main__":
    # Example usage for testing
    agent = Group14()
    agent.initialize()
    print(agent.propose(None))
    print(agent.accept({"offer": "example_offer"}))
