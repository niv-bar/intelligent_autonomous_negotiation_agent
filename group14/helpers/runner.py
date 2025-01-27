from group14.group14 import Group14

import logging
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def save_results(results, filename="data/results.csv"):
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["scenario", "total_utility", "average_utility"])
        writer.writeheader()
        writer.writerows(results)


def run_negotiation():
    """
    Simulate a negotiation session with the Group14 agent.
    """
    agent = Group14()
    agent.initialize()

    opponent_offers = [
        {"price": 90, "quality": "medium"},
        {"price": 85, "quality": "medium"},
        {"price": 80, "quality": "high"},
    ]
    time_steps = [0.7, 0.4, 0.1]  # Remaining time fractions

    for i, offer in enumerate(opponent_offers):
        time_remaining = time_steps[i]
        logging.info(f"Time remaining: {time_remaining}")
        logging.info(f"Opponent's offer: {offer}")

        counter_offer = agent.propose(offer, time_remaining=time_remaining)
        logging.info(f"Agent's counter-offer: {counter_offer}")

        accepted = agent.accept(offer, time_remaining=time_remaining)
        logging.info(f"Offer accepted: {accepted}")


def simulate_scenarios():
    """
    Simulate multiple negotiation scenarios with varying opponent behavior.
    """
    scenarios = [
        {
            "offers": [{"price": 95, "quality": "medium"}, {"price": 85, "quality": "medium"}, {"price": 75, "quality": "high"}],
            "time_steps": [0.9, 0.6, 0.2],
        },
        {
            "offers": [{"price": 100, "quality": "low"}, {"price": 80, "quality": "medium"}, {"price": 70, "quality": "high"}],
            "time_steps": [0.8, 0.5, 0.1],
        },
        {
            "offers": [{"price": 90, "quality": "medium"}, {"price": 70, "quality": "medium"}, {"price": 50, "quality": "high"}],
            "time_steps": [0.7, 0.3, 0.05],
        },
    ]

    results = []

    for idx, scenario in enumerate(scenarios):
        logging.info(f"--- Scenario {idx + 1} ---")
        agent = Group14()
        agent.initialize()

        for i, offer in enumerate(scenario["offers"]):
            time_remaining = scenario["time_steps"][i]
            logging.info(f"Time remaining: {time_remaining}")
            logging.info(f"Opponent's offer: {offer}")

            counter_offer = agent.propose(offer, time_remaining=time_remaining)
            logging.info(f"Agent's counter-offer: {counter_offer}")

            accepted = agent.accept(offer, time_remaining=time_remaining)
            logging.info(f"Offer accepted: {accepted}")

        logging.info(f"Final Metrics: {agent.get_metrics()}")

        metrics = agent.get_metrics()
        results.append({"scenario": f"Scenario {idx + 1}", "total_utility": metrics["total_utility"], "average_utility": metrics["average_utility"]})

    save_results(results)
    logging.info("Results saved to results.csv")


if __name__ == "__main__":
    simulate_scenarios()
