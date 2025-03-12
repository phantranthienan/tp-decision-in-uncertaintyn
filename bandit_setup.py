import numpy as np
import random

def pull(arm, success_rates):
    return 1 if random.random() < success_rates[arm] else 0  # Bernoulli distribution

def initialize_bandit(K, success_rates):
    s = np.zeros(K)  # Cumulative rewards
    n = np.ones(K)   # Pull counts
    total_reward = 0

    for i in range(K):
        r = pull(i, success_rates)
        s[i] += r
        total_reward += r

    return s, n, total_reward