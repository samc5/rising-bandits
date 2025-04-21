import math
import random
import numpy as np


class BanditProblem:
    def __init__(self, k, init_params, reward_func):
        self.k = k
        self.pull_counts = [0] * k
        self.total_rewards = [0] * k
        self.params = init_params
        self.reward_func = reward_func
        self.external_regret = 0

    def get_reward(self, reward_func, param_map, t, verbose = None):
        res = reward_func(param_map, t)
        if verbose is not None:
            print(f"Reward at time {t}: {res}")
        return res

    def get_all_rewards(self, reward_func, t):
        rewards = []
        for i in range(self.k):
            reward = self.get_reward(reward_func, self.params[i], t)
            rewards.append(reward)
        return rewards

    def get_external_optimal(self):
        rewards = self.get_all_rewards(self.reward_func, 0)
        return np.argmax(np.array(rewards))

    def pull_arm(self, i, verbose = None):
        self.pull_counts[i] += 1
        reward = self.get_reward(self.reward_func, self.params[i], self.pull_counts[i], verbose)
        self.total_rewards[i] += reward
        external_optimal = self.get_external_optimal()
        self.external_regret += (self.total_rewards[external_optimal] - reward)

    def print_stats(self):
        print("Total rewards: ", self.total_rewards)
        print("Pull counts: ", self.pull_counts)
        print("External regret: ", self.external_regret)

def exponential_reward(param_map, t):
        a = param_map['a']
        c = param_map['c']
        return c * (1 - math.exp(-1 * a * t))



def polynomial_reward(param_map, t):
        p = param_map['p']
        b = param_map['b']
        c = param_map['c']
        inner = t + (b ** (1/p))
        inner2 = inner ** (-1 * p)
        inner3 = 1 - b * inner2
        return c * inner3



# real tests
print(exponential_reward({'a': 0.5, 'c': 0.5}, 150))
print(polynomial_reward({'p': 0.5, 'b': 0.6, 'c': 0.9}, 1503355))

# Two-armed bandit test
problem = BanditProblem(2, [{'a': 0.01, 'c': 0.5}, {'a': 0.005, 'c': 1}], exponential_reward)
for i in range(100):
    arm = random.randint(0, 1)
    problem.pull_arm(arm, verbose = True)
problem.print_stats()
for i in range(1000):
    arm = random.randint(0, 1)
    problem.pull_arm(arm, verbose = True)
problem.print_stats()