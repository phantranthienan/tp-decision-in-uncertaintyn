import time
import numpy as np

from bandit_setup import pull, initialize_bandit

from bandit_algos import (
    bandit_random, 
    bandit_epsilon_greedy, 
    bandit_epsilon_greedy_decay, 
    bandit_ucb
)

def run_bandit(K, N, success_rates, algo):
    s, n, total_reward = initialize_bandit(K, success_rates)

    for t in range(K, N):
        chosen_arm = algo(s, n, t)

        r = pull(chosen_arm, success_rates)
        s[chosen_arm] += r
        n[chosen_arm] += 1
        total_reward += r
    
    return total_reward

def run_scenario(K, N, success_rates, num_runs):
    random_algo = bandit_random
    eps_algo = bandit_epsilon_greedy(epsilon=0.1)
    eps_decay_algo = bandit_epsilon_greedy_decay
    ucb_algo = bandit_ucb

    print(f"Scenario:")
    print(f"K={K}")
    print(f"N={N}")
    print(f"success_rates={success_rates}")
    print(f"num_runs={num_runs}")

    print("Running random...")
    r_random, t_random = average_runs(K, N, success_rates, random_algo, num_runs)

    print("Running e-greedy (eps=0.1)...")
    r_eps, t_eps = average_runs(K, N, success_rates, eps_algo, num_runs)

    print("Running e-greedy decay...")
    r_epsdec, t_epsdec = average_runs(K, N, success_rates, eps_decay_algo, num_runs)

    print("Running UCB...")
    r_ucb, t_ucb = average_runs(K, N, success_rates, ucb_algo, num_runs)

    results = {
        "random": (r_random, t_random),
        "e-greedy": (r_eps, t_eps),
        "e-greedy-decay": (r_epsdec, t_epsdec),
        "UCB": (r_ucb, t_ucb)
    }

    return results

def average_runs(K, N, success_rates, algo, num_runs=10):
    all_rewards = np.zeros((num_runs, N))
    all_times   = np.zeros((num_runs, N))
    
    for i in range(num_runs):
        print("Run", i+1, "of", num_runs)
        rewards_i, times_i = run_bandit_trace(K, N, success_rates, algo)
        all_rewards[i, :] = rewards_i
        all_times[i, :] = times_i
    
    mean_rewards = np.mean(all_rewards, axis=0)
    mean_times = np.mean(all_times, axis=0)
    return mean_rewards, mean_times

def run_bandit_trace(K, N, success_rates, algo):
    s, n, total_reward = initialize_bandit(K, success_rates)
    
    cumulative_rewards = [0] * N
    cumulative_times = [0] * N

    for t in range(K):
        cumulative_rewards[t] = total_reward
        cumulative_times[t] = 0

    start_time = time.time()

    for t in range(K, N):
        chosen_arm = algo(s, n, t)
        r = pull(chosen_arm, success_rates)
        s[chosen_arm] += r
        n[chosen_arm] += 1
        total_reward += r
        
        t_after = time.time()
        
        cumulative_rewards[t] = total_reward

        cumulative_times[t] = (t_after - start_time)

    return cumulative_rewards, cumulative_times