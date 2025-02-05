import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend to avoid the error
import matplotlib.pyplot as plt
import networkx as nx


# Create a directed graph
G = nx.DiGraph()

# Define the main steps of the negotiation process
nodes = {
    "Start": "Start Negotiation",
    "Receive": "Receive Offer",
    "Check_UF": "Check Utility Function",
    "Update_RV": "Update Partner's Reservation Value",
    "Acceptance": "Acceptance Strategy",
    "Accept": "Accept Offer",
    "Reject": "Reject Offer & Generate Counteroffer",
    "Send_Counter": "Send Counteroffer",
    "End": "End Negotiation"
}

# Add edges representing the process flow
edges = [
    ("Start", "Receive"),
    ("Receive", "Check_UF"),
    ("Check_UF", "Update_RV"),
    ("Check_UF", "Acceptance"),
    ("Acceptance", "Accept"),
    ("Acceptance", "Reject"),
    ("Reject", "Send_Counter"),
    ("Send_Counter", "Receive"),  # Loop back to receiving an offer
    ("Check_UF", "End")  # Negotiation ends if no further action is possible
]

# Add nodes and edges to the graph
G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

# Define positions for a better layout
pos = nx.spring_layout(G, seed=42)

# Create the plot
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=9,
        labels={key: nodes[key] for key in nodes})

plt.title("Negotiation Process Flowchart")

# Show the plot
plt.show()
