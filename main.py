import numpy as np
import matplotlib.pyplot as plt

from bandit_algos import (
    bandit_random, 
    bandit_epsilon_greedy, 
    bandit_epsilon_greedy_decay, 
    bandit_ucb
)
from bandit_run import run_bandit_trace

def average_runs(K, N, success_rates, algo, num_runs=10):
    print("Running", algo.__name__)
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

def main():
    K = 9
    N = 100000
    success_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_runs = 1

    random_algo = bandit_random
    eps_algo = bandit_epsilon_greedy(epsilon=0.1)
    eps_decay_algo = bandit_epsilon_greedy_decay
    ucb_algo = bandit_ucb

    print("Running random...")
    r_random, t_random = average_runs(K, N, success_rates, random_algo, num_runs)
    print("Running e-greedy (eps=0.1)...")
    r_eps, t_eps = average_runs(K, N, success_rates, eps_algo, num_runs)
    print("Running e-greedy decay...")
    r_epsdec, t_epsdec = average_runs(K, N, success_rates, eps_decay_algo, num_runs)
    print("Running UCB...")
    r_ucb, t_ucb = average_runs(K, N, success_rates, ucb_algo, num_runs)

    # Plotting
    step_indices = np.linspace(0, N-1, 10, dtype=int)

    plt.figure(figsize=(16,8))

    # Cumulative Reward
    plt.subplot(1, 2, 1)
    plt.plot(step_indices, r_ucb[step_indices], label='UCB', marker='o')
    plt.plot(step_indices, r_epsdec[step_indices], label='e-greedy-dec', marker='x')
    plt.plot(step_indices, r_eps[step_indices], label='e-greedy', marker='^')
    plt.plot(step_indices, r_random[step_indices], label='random', marker='s')
    plt.title("Cumulative reward")
    plt.xlabel("Number of pulls")
    plt.ylabel("Cumulative reward")
    plt.legend()

    # Execution Time
    plt.subplot(1, 2, 2)
    plt.plot(step_indices, t_ucb[step_indices], label='UCB', marker='o')
    plt.plot(step_indices, t_epsdec[step_indices], label='e-greedy-dec', marker='x')
    plt.plot(step_indices, t_eps[step_indices], label='e-greedy', marker='^')
    plt.plot(step_indices, t_random[step_indices], label='random', marker='s')
    plt.title("Time (in seconds)")
    plt.xlabel("Number of pulls")
    plt.ylabel("Time (seconds)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()