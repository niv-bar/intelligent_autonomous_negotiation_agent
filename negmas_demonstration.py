from negmas import (
    SAOMechanism,
    AspirationNegotiator,
    LinearUtilityFunction,
    make_issue
)


def create_simple_negotiation():
    """
    Creates and runs a simple negotiation between two agents over a single issue.
    """
    # Define the negotiation issue with discrete values
    possible_prices = list(range(10, 101, 5))  # [10, 15, 20, ..., 95, 100]
    issues = [make_issue(name="price", values=possible_prices)]

    # Create utility functions for both agents
    # For seller: higher price is better (weights are positive)
    seller_ufun = LinearUtilityFunction(
        weights={"price": 1.0},
        issues=issues
    )

    # For buyer: lower price is better (weights are negative)
    buyer_ufun = LinearUtilityFunction(
        weights={"price": -1.0},
        issues=issues
    )

    # Create negotiators with their utility functions
    seller = AspirationNegotiator(name="seller", ufun=seller_ufun)
    buyer = AspirationNegotiator(name="buyer", ufun=buyer_ufun)

    # Create the negotiation mechanism
    mechanism = SAOMechanism(issues=issues, n_steps=20)

    # Add both negotiators to the mechanism
    mechanism.add(seller)
    mechanism.add(buyer)

    # Run the negotiation
    result = mechanism.run()

    # Print results
    if result.agreement is None:
        print("No agreement reached")
    else:
        print(f"Agreement reached: {result.agreement}")
        print(f"Steps taken: {result.step}")
        print(f"Final price: ${result.agreement[0]}")
        print(f"Seller utility: {seller_ufun(result.agreement):.2f}")
        print(f"Buyer utility: {buyer_ufun(result.agreement):.2f}")


if __name__ == "__main__":
    create_simple_negotiation()