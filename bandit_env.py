import numpy as np

def pull(arm, success_rates):
    return np.random.binomial(1, success_rates[arm])

def initialize_bandit(K, success_rates):
    s = np.zeros(K)  # Cumulative rewards
    n = np.ones(K)   # Pull counts
    total_reward = 0

    for i in range(K):
        r = pull(i, success_rates)
        s[i] += r
        total_reward += r

    return s, n, total_reward