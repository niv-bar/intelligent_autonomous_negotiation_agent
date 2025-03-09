import pandas as pd
import matplotlib.pyplot as plt


# Load the results file
file_path = "all_scores.csv"
df = pd.read_csv(file_path)

# First plot - Average Utility
group14_data = df[df["strategy"] == "Group14"]
avg_utility_per_agent = df.groupby("strategy")["utility"].mean().sort_values()
plt.figure(figsize=(10, 6))
colors = ["blue" if agent != "Group14" else "red" for agent in avg_utility_per_agent.index]
avg_utility_per_agent.plot(kind="bar", color=colors, alpha=0.7)
plt.axhline(group14_data["utility"].mean(), color="red", linestyle="--", label="Group14 Avg Utility")
plt.title("Average Utility per Agent")
plt.xlabel("Agent Strategy")
plt.ylabel("Utility")
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Second plot - Utility Comparison Across Scenarios
avg_utility_per_scenario = df.groupby("scenario")["utility"].mean()
agents_avg_utility_per_scenario = df.groupby(["scenario", "strategy"])["utility"].mean().unstack()
plt.figure(figsize=(12, 6))
for agent in agents_avg_utility_per_scenario.columns:
    color = "blue" if agent != "Group14" else "red"
    linestyle = "--" if agent != "Group14" else "-"
    alpha = 0.4 if agent != "Group14" else 0.8
    plt.plot(agents_avg_utility_per_scenario.index, agents_avg_utility_per_scenario[agent],
             marker="o", linestyle=linestyle, color=color, label=agent, alpha=alpha)
plt.title("Utility Comparison Across Scenarios")
plt.xlabel("Scenario")
plt.ylabel("Utility")
plt.legend(loc="upper right", bbox_to_anchor=(0.16, 0.35))
plt.grid(True)

# Third plot - Utility Progress Over Time
df_time_limited = df[df["time"] <= 50]
agents_avg_utility_over_time = df_time_limited.groupby(["time", "strategy"])["utility"].mean().unstack()
plt.figure(figsize=(12, 6))
for agent in agents_avg_utility_over_time.columns:
    color = "blue" if agent != "Group14" else "red"
    linestyle = "--" if agent != "Group14" else "-"
    alpha = 0.4 if agent != "Group14" else 0.8
    plt.plot(agents_avg_utility_over_time.index, agents_avg_utility_over_time[agent],
             marker="o", linestyle=linestyle, color=color, label=agent, alpha=alpha)
plt.title("Utility Progress Over Time")
plt.xlabel("Negotiation Time")
plt.ylabel("Utility")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.grid(True)

# plt.show()