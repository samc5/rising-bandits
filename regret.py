import pandas as pd
import numpy as np
import json
import math
from collections import Counter
import os

horizon_to_analyze = 35000
n_arms_to_analyze = 35

print("Loading configuration for analysis...")
try:

    epsilon_used = 0.55
    n_arms = n_arms_to_analyze
    horizon = horizon_to_analyze
    print(f"Using: n_arms={n_arms}, horizon={horizon}, epsilon={epsilon_used}")

    print("Determining player list used in simulation...")
    youths_temp = pd.read_csv("youths.csv") 
    player_results_temp = {}
    for i, row in youths_temp.iterrows():
        rewards_list = player_results_temp.get(row['batter'], [])
        rewards_list.append(row['reward'])
        player_results_temp[row['batter']] = rewards_list
    all_eligible_players_temp = list(player_results_temp.keys())
    if len(all_eligible_players_temp) < n_arms: exit("Error: Not enough players in youths.csv")
    players = all_eligible_players_temp[:n_arms]
    print(f"Loaded/Determined {len(players)} players.")

except FileNotFoundError as e:
    exit(f"Error: Config or player file not found: {e}")
except KeyError as e:
    exit(f"Error: Missing parameter {e} in config file.")
except Exception as e:
    exit(f"Error loading configuration: {e}")

print("Loading youths.csv for analysis...")
youths = pd.read_csv("youths.csv")


def get_final_windowed_avg(player_id, data, epsilon):
    player_data = data[data['batter'] == player_id]
    if player_data.empty: return np.nan
    if 'trial' not in player_data.columns: return np.nan
    N = player_data['trial'].max()
    if N is None or pd.isna(N) or N <= 0: return np.nan
    h = math.floor(epsilon * N)
    if 'reward' not in player_data.columns: return np.nan
    if h <= 0:
        if N > 0: return player_data['reward'].mean()
        else: return np.nan
    last_h_data = player_data[player_data['trial'] > N - h]
    if last_h_data.empty:
         if N > 0: return player_data['reward'].mean()
         else: return np.nan
    if 'reward' in last_h_data.columns and not last_h_data['reward'].empty:
        return last_h_data['reward'].mean()
    else: return np.nan

# --- Calculate 'true' final windowed performance for all arms ---
true_final_perf = {}
print("\nCalculating 'True' Final Windowed Performance (based on youths.csv):")
valid_perf_calculated = False
for i, player_id in enumerate(players):
    perf = get_final_windowed_avg(player_id, youths, epsilon_used)
    if not pd.isna(perf):
        true_final_perf[i] = perf
        print(f"  Player {i} ({player_id}): {perf:.4f}")
        valid_perf_calculated = True
    else:
        print(f"  Player {i} ({player_id}): Could not calculate valid performance.")
        true_final_perf[i] = -np.inf
if not valid_perf_calculated: exit("Error: No valid true performance calculated.")

# --- Find the optimal performance based on this metric ---
optimal_player_index = max(true_final_perf, key=true_final_perf.get)
optimal_final_window_reward = true_final_perf[optimal_player_index]
if optimal_final_window_reward == -np.inf: exit("Error: Optimal performance is invalid.")
print(f"\nOptimal Player Index (based on final window): {optimal_player_index} ({players[optimal_player_index]})")
print(f"Optimal Final Windowed Reward (mu_star_windowed): {optimal_final_window_reward:.4f}")

results_file_path = f"experiments/exp_baseball/{horizon_to_analyze}.json" # Path based on analyzed horizon
agent_name = "\\succrejectshor t" # Agent key used in results dictionary
try:
    print(f"\nLoading simulation results from: {results_file_path}")
    with open(results_file_path, 'r') as f:
        simulation_results = json.load(f)
    if agent_name not in simulation_results: raise KeyError(f"Agent {agent_name} not found")
    recommendations = simulation_results[agent_name]["recommendations"]

except FileNotFoundError: exit(f"Error: Results file not found: {results_file_path}")
except KeyError as e: exit(f"Error: Missing key {e} in results file.")
except Exception as e: exit(f"Error loading results: {e}")

# Calculate Simple Regret
simple_regrets = []
valid_trials_for_regret = 0
print("\nCalculating Simple Regret per Trial:")
for trial_idx, rec_idx_float in enumerate(recommendations):
    try: rec_idx = int(rec_idx_float)
    except (ValueError, TypeError): continue
    if not (0 <= rec_idx < n_arms): continue
    recommended_final_window_reward = true_final_perf.get(rec_idx, None)
    if recommended_final_window_reward is None or recommended_final_window_reward == -np.inf : continue
    regret = optimal_final_window_reward - recommended_final_window_reward
    simple_regrets.append(regret)
    valid_trials_for_regret += 1

if simple_regrets:
    average_simple_regret = np.mean(simple_regrets)
    std_dev_simple_regret = np.std(simple_regrets)
    print(f"\nAverage Simple Regret across {valid_trials_for_regret} valid trials: {average_simple_regret:.4f}")
    print(f"Std Dev Simple Regret across {valid_trials_for_regret} valid trials: {std_dev_simple_regret:.4f}")
else:
    print("\nCould not calculate simple regret for any valid trial.")


print("\nRecommendation Counts (Player Index):")
valid_recommendations = [int(r) for r in recommendations if isinstance(r, (int, float)) and 0 <= int(r) < n_arms]
recommendation_counts = Counter(valid_recommendations)
for idx, count in recommendation_counts.most_common():
    if 0 <= idx < len(players): player_name = players[idx]
    else: player_name = "Unknown Player"
    print(f"  Player Index {idx} ({player_name}): {count} / {len(valid_recommendations)} valid trials")

print("\nAnalysis script finished.")

print("\n--- Starting Cumulative Regret Calculations ---")


# Benchmark A: Best Overall Average Reward from youths.csv
print("Calculating Overall Average Rewards...")
overall_avg_perf = {}
valid_overall_avg_calculated = False
for i, player_id in enumerate(players):
    player_data = youths[youths['batter'] == player_id]
    if not player_data.empty and 'reward' in player_data.columns and not player_data['reward'].empty:
        avg_rew = player_data['reward'].mean()
        overall_avg_perf[i] = avg_rew
        print(f"  Player {i} ({player_id}) Overall Avg: {avg_rew:.4f}")
        valid_overall_avg_calculated = True
    else:
        print(f"  Player {i} ({player_id}): Could not calculate overall average.")
        overall_avg_perf[i] = -np.inf

if not valid_overall_avg_calculated:
    print("Error: Could not calculate overall average for any player.")

else:
    optimal_overall_player_index = max(overall_avg_perf, key=overall_avg_perf.get)
    optimal_overall_avg_reward = overall_avg_perf[optimal_overall_player_index] # mu_star_overall
    if optimal_overall_avg_reward == -np.inf:
        print("Error: Optimal overall average performance could not be determined.")
    else:
        print(f"\nBest Overall Avg Player Index: {optimal_overall_player_index} ({players[optimal_overall_player_index]})")
        print(f"Best Overall Avg Reward (mu_star_overall): {optimal_overall_avg_reward:.4f}")

# Benchmark B: Best Final Windowed Average (already calculated)
print(f"\nBest Final Windowed Reward (mu_star_windowed): {optimal_final_window_reward:.4f} (Player {optimal_player_index})")



static_regrets_overall_avg = []
static_regrets_final_window = []

# Check if 'scores' key exists and is a list
if "scores" not in simulation_results[agent_name] or not isinstance(simulation_results[agent_name]["scores"], list):
     print(f"Error: 'scores' data not found or not in expected list format for agent {agent_name}.")
else:
    all_trial_scores = simulation_results[agent_name]["scores"]
    n_trials_loaded = len(all_trial_scores)
    print(f"\nCalculating cumulative regrets for {n_trials_loaded} trials...")

    for trial_idx, trial_scores in enumerate(all_trial_scores):
        # Ensure trial_scores is a list or array of numbers
        if not isinstance(trial_scores, (list, np.ndarray)):
            print(f"Warning: Scores for trial {trial_idx} are not in list/array format. Skipping trial.")
            continue

        valid_scores = [s for s in trial_scores if not pd.isna(s)]
        if len(valid_scores) != horizon:
             pass 

        cumulative_reward_algo = np.sum(valid_scores)

        # Calculate Static Regret (Proxy A: vs Best Overall Average)
        if optimal_overall_avg_reward != -np.inf:
            regret_overall = (horizon * optimal_overall_avg_reward) - cumulative_reward_algo
            static_regrets_overall_avg.append(regret_overall)
        else:
            static_regrets_overall_avg.append(np.nan) 
        # Calculate Static Regret (Proxy B: vs Best Final Windowed Average)
        if optimal_final_window_reward != -np.inf:
            regret_final_window = (horizon * optimal_final_window_reward) - cumulative_reward_algo
            static_regrets_final_window.append(regret_final_window)
        else:
             static_regrets_final_window.append(np.nan)

    print("\n--- Average Cumulative Regret Results ---")

    if static_regrets_overall_avg:
        avg_static_regret_overall = np.nanmean(static_regrets_overall_avg) # Use nanmean to ignore NaNs
        std_static_regret_overall = np.nanstd(static_regrets_overall_avg)
        print(f"\nStatic/External Regret (vs Best Overall Avg Reward):")
        print(f"  Average: {avg_static_regret_overall:.2f}")
        print(f"  Std Dev: {std_static_regret_overall:.2f}")
    else:
        print("\nCould not calculate Static Regret (vs Best Overall Avg).")

    if static_regrets_final_window:
        avg_static_regret_final = np.nanmean(static_regrets_final_window)
        std_static_regret_final = np.nanstd(static_regrets_final_window)
        print(f"\nStatic Regret (vs Best Final Windowed Avg Reward):")
        print(f"  Average: {avg_static_regret_final:.2f}")
        print(f"  Std Dev: {std_static_regret_final:.2f}")
    else:
        print("\nCould not calculate Static Regret (vs Best Final Window).")
