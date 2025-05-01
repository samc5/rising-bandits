import pandas as pd
import numpy as np
import json
import math
from collections import Counter
import os
import matplotlib.pyplot as plt
import time 
from environment.environment import SRBEnvironment 
from agent.agent import * 
from runner.runner import Runner 
from tqdm.auto import tqdm 


horizons_to_test = [5000, 15000, 25000, 35000] 
n_arms_to_test = [10, 20, 30, 35]       
epsilon_used = 0.55                  
n_trials_per_config = 5 

results_base_path = "experiments/exp_baseball_regret_sweep"

youths = pd.read_csv("youths.csv")


player_results_full = {}
for i, row in youths.iterrows():
    rewards_list = player_results_full.get(row['batter'], [])
    rewards_list.append(row['reward'])
    player_results_full[row['batter']] = rewards_list
all_eligible_players_full = list(player_results_full.keys())
max_available_arms = len(all_eligible_players_full)
print(f"Total eligible players available: {max_available_arms}")

n_arms_to_test = [n for n in n_arms_to_test if n <= max_available_arms]

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

results_summary = [] #
total_runs = len(horizons_to_test) * len(n_arms_to_test)
current_run = 0
start_time_total = time.time()

for horizon in horizons_to_test:
    for n_arms in n_arms_to_test:
        current_run += 1
        start_time_run = time.time()
        print(f"\nRunning Configuration {current_run}/{total_runs}: Horizon={horizon}, N_Arms={n_arms}")

        players = all_eligible_players_full[:n_arms]
        if len(players) != n_arms:
             print(f"Warning: Could only select {len(players)} players for n_arms={n_arms}. Skipping config.")
             continue

        param_agent_sr_srb = {
            "n_arms": n_arms,
            "horizon": horizon,
            "eps": epsilon_used
        }

        def safe_get(lst, idx):
            try: return lst[int(idx)]
            except (IndexError, ValueError, TypeError): return None # Return None on failure
        arms = [
            (lambda player_id: lambda x: safe_get(player_results_full.get(player_id, []), x))(pid)
            for pid in players
        ]

        try:
            env = SRBEnvironment(horizon=horizon, actions=arms, names=players, noise=0)
            agent = SRSrb(**param_agent_sr_srb) 

            runner = Runner(
                environment=env,
                agent=agent,
                n_trials=n_trials_per_config,
                horizon=horizon,
                n_actions=n_arms,
                actions=arms, 
                log_path=results_base_path
            )
        except Exception as e:
            print(f"Error setting up MAB components for H={horizon}, K={n_arms}: {e}. Skipping config.")
            continue
        #Sims
        print(f"Running {n_trials_per_config} trials...")
        try:
            runner.perform_simulations()
            simulation_results_current = runner.results 

            print("Simulation complete.")
        except Exception as e:
            print(f"Error during simulation for H={horizon}, K={n_arms}: {e}. Skipping config.")
            continue

        try:
            # Calculate True Performance Benchmarks (for the current 'players' subset)
            true_final_perf = {}
            overall_avg_perf = {}
            valid_perf_calculated = False
            valid_overall_avg_calculated = False

            for i, player_id in enumerate(players):
                # Final Windowed
                perf = get_final_windowed_avg(player_id, youths, epsilon_used)
                if not pd.isna(perf):
                    true_final_perf[i] = perf
                    valid_perf_calculated = True
                else:
                    true_final_perf[i] = -np.inf
                # Overall Average
                player_data = youths[youths['batter'] == player_id]
                if not player_data.empty and 'reward' in player_data.columns and not player_data['reward'].empty:
                    avg_rew = player_data['reward'].mean()
                    overall_avg_perf[i] = avg_rew
                    valid_overall_avg_calculated = True
                else:
                    overall_avg_perf[i] = -np.inf

            # Determine Optimal Benchmarks
            optimal_final_window_reward = -np.inf
            if valid_perf_calculated:
                optimal_player_index_window = max(true_final_perf, key=true_final_perf.get)
                optimal_final_window_reward = true_final_perf[optimal_player_index_window]

            optimal_overall_avg_reward = -np.inf
            if valid_overall_avg_calculated:
                optimal_player_index_overall = max(overall_avg_perf, key=overall_avg_perf.get)
                optimal_overall_avg_reward = overall_avg_perf[optimal_player_index_overall]

            # Calculate Regrets
            simple_regrets = []
            static_regrets_overall_avg = []
            static_regrets_final_window = []

            agent_key = agent.name # Get agent name used as key in results
            if agent_key not in simulation_results_current or "scores" not in simulation_results_current[agent_key]:
                 print(f"Warning: Scores not found for agent '{agent_key}'. Skipping regret calculation.")
            else:
                all_trial_scores = simulation_results_current[agent_key]["scores"]
                recommendations = simulation_results_current[agent_key]["recommendations"]

                for trial_idx, trial_scores in enumerate(all_trial_scores):
                    if not isinstance(trial_scores, (list, np.ndarray)): continue
                    valid_scores = [s for s in trial_scores if not pd.isna(s)]
                    cumulative_reward_algo = np.sum(valid_scores)

                    # Simple Regret
                    rec_idx = int(recommendations[trial_idx]) if trial_idx < len(recommendations) and recommendations[trial_idx] is not None else -1
                    if 0 <= rec_idx < n_arms:
                         rec_perf = true_final_perf.get(rec_idx, -np.inf)
                         if optimal_final_window_reward != -np.inf and rec_perf != -np.inf:
                              simple_regrets.append(optimal_final_window_reward - rec_perf)
                         else: simple_regrets.append(np.nan)
                    else: simple_regrets.append(np.nan)


                    # Static Regret (Overall Avg)
                    if optimal_overall_avg_reward != -np.inf:
                        static_regrets_overall_avg.append((horizon * optimal_overall_avg_reward) - cumulative_reward_algo)
                    else: static_regrets_overall_avg.append(np.nan)

                    # Static Regret (Final Window)
                    if optimal_final_window_reward != -np.inf:
                        static_regrets_final_window.append((horizon * optimal_final_window_reward) - cumulative_reward_algo)
                    else: static_regrets_final_window.append(np.nan)

            results_summary.append({
                "horizon": horizon,
                "n_arms": n_arms,
                "avg_simple_regret": np.nanmean(simple_regrets) if simple_regrets else np.nan,
                "avg_static_regret_overall": np.nanmean(static_regrets_overall_avg) if static_regrets_overall_avg else np.nan,
                "avg_static_regret_final_window": np.nanmean(static_regrets_final_window) if static_regrets_final_window else np.nan,
                "std_simple_regret": np.nanstd(simple_regrets) if simple_regrets else np.nan,
                "std_static_regret_overall": np.nanstd(static_regrets_overall_avg) if static_regrets_overall_avg else np.nan,
                "std_static_regret_final_window": np.nanstd(static_regrets_final_window) if static_regrets_final_window else np.nan
            })
            print("Analysis complete for this configuration.")

        except Exception as e:
            continue 

results_df = pd.DataFrame(results_summary)
print("\nResults Summary DataFrame:")
print(results_df)

plt.style.use('seaborn-v0_8-darkgrid') 
fig1, ax1 = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
regret_types = [
    ('avg_simple_regret', 'Simple Regret (vs Best Final Window)'),
    ('avg_static_regret_overall', 'Static Regret (vs Best Overall Avg)'),
    ('avg_static_regret_final_window', 'Static Regret (vs Best Final Window Avg)')
]
for i, (regret_col, title) in enumerate(regret_types):
    for n_arms_val in n_arms_to_test:
        subset = results_df[results_df['n_arms'] == n_arms_val].sort_values('horizon')
        if not subset.empty:
             ax1[i].plot(subset['horizon'], subset[regret_col], marker='o', linestyle='-', label=f'N_Arms = {n_arms_val}')
    ax1[i].set_ylabel('Average Regret')
    ax1[i].set_title(title)
    ax1[i].legend()
    ax1[i].grid(True)
ax1[-1].set_xlabel('Horizon (T)')
fig1.suptitle('Regret vs Horizon for Different Numbers of Arms', fontsize=16, y=0.99)
fig1.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('regret_vs_horizon.png') 
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
for i, (regret_col, title) in enumerate(regret_types):
    for horizon_val in horizons_to_test:
        subset = results_df[results_df['horizon'] == horizon_val].sort_values('n_arms')
        if not subset.empty:
             ax2[i].plot(subset['n_arms'], subset[regret_col], marker='o', linestyle='-', label=f'Horizon = {horizon_val}')
    ax2[i].set_ylabel('Average Regret')
    ax2[i].set_title(title)
    ax2[i].legend()
    ax2[i].grid(True)
ax2[-1].set_xlabel('Number of Arms (K)')
fig2.suptitle('Regret vs Number of Arms for Different Horizons', fontsize=16, y=0.99)
fig2.tight_layout(rect=[0, 0.03, 1, 0.97]) 
plt.savefig('regret_vs_n_arms.png') 