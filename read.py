import pandas as pd
import numpy as np
import json
from environment.environment import SRBEnvironment
from agent.agent import *
from runner.runner import Runner
from collections import Counter

data = pd.read_csv("youths.csv")
player_results = {}
for i, row in data.iterrows():
    if i % 10000 == 0:
        print(i)
    rewards_list = player_results.get(row['batter'], [])
    rewards_list.append(row['reward'])
    player_results[row['batter']] = rewards_list
n_arms = 35
horizon = 1000 * n_arms

players = list(player_results.keys())[:n_arms]

def safe_get(lst, idx):
    try:
        return lst[int(idx)]
    except (IndexError, ValueError, TypeError):
        return np.mean(np.array(lst)) 

arms = [
    (lambda i: lambda x: safe_get(player_results[i], x))(i)
    for i in players
]
print(arms)
print(len(arms))
for i in range(10):
    print(arms[0](1))
"""
    "agent_ucb_srb": {
        "n_arms": 7,
        "exp_param": 57.12041528623219,
        "horizon": 3000,
        "eps": 0.25,
        "sigma": 0.05
    },
"""
# Build up the blocks
param_agent_sr_srb =     {
        "n_arms": len(arms),
        "horizon": horizon,
        "eps": 0.55
    }
param_agent_ucb_srb = {
        "n_arms": len(arms),
        "exp_param": 57.12041528623219,
        "horizon": horizon,
        "eps": 0.25,
        "sigma": 0.05
}
param_env = {'horizon': horizon}
env = SRBEnvironment(horizon=horizon, actions=arms, names=players, noise=1)


agent = SRSrb(**param_agent_sr_srb)


runner = Runner(
    environment=env,
    agent=None,
    n_trials=10,
    horizon=horizon,
    n_actions=len(arms),
    actions=arms,
    log_path="experiments/exp_baseball"
)

print("\n################# T = " + str(horizon) + " #################")
    # assign the new agent
runner.agent = SRSrb(**param_agent_sr_srb)
# runner.agent = UcbSRB(**param_agent_ucb_srb)
# perform simulation
runner.perform_simulations()
recommendations = runner.results[agent.name]["recommendations"]
estimators_all_trials = np.array(runner.results[agent.name]["estimators"])

print(f"Players: {players}")
# print("Recommendation Raw List: " + str(recommendations)) # Optional: raw list



recommendation_counts = Counter(recommendations)
print("\nRecommendation Counts (Player Index):")
# Sort by frequency for clarity
for idx, count in recommendation_counts.most_common():
    player_name = players[int(idx)] if 0 <= int(idx) < len(players) else f"Unknown Index {int(idx)}"
    print(f"  Player Index {int(idx)} ({player_name}): {count} / {runner.n_trials} trials")

avg_final_estimators = np.mean(estimators_all_trials, axis=0)
print("\nAverage Final Estimator Value (mu_hat) per Player across trials:")
# Create a list of tuples (player_index, player_id, avg_estimator) for sorting
player_avg_estimators = []
for i, player_id in enumerate(players):
     if i < len(avg_final_estimators): # Check bounds
        player_avg_estimators.append((i, player_id, avg_final_estimators[i]))

# Sort players by their average final estimated value (descending)
player_avg_estimators.sort(key=lambda item: item[2], reverse=True)

for idx, player_id, avg_est in player_avg_estimators:
    print(f"  Player {idx} ({player_id}): {avg_est:.4f}")

# Identify player most often recommended
if recommendation_counts: # Check if there are any recommendations
    most_common_rec_idx = recommendation_counts.most_common(1)[0][0]
    most_common_player = players[int(most_common_rec_idx)] if 0 <= int(most_common_rec_idx) < len(players) else "Unknown"
    print(f"\nMost Frequently Recommended Player Index: {int(most_common_rec_idx)} ({most_common_player})")
    if 0 <= int(most_common_rec_idx) < len(avg_final_estimators):
       print(f"  Their Average Final Estimator: {avg_final_estimators[int(most_common_rec_idx)]:.4f}")
else:
    print("\nNo recommendations were made (or recorded properly).")

runner.save_output(str(horizon))
print("##############################################################")