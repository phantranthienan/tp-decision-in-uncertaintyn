import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, N):
    step_indices = np.linspace(0, N-1, 10, dtype=int)

    plt.figure(figsize=(16,8))

    r_ucb, t_ucb = results['UCB']
    r_epsdec, t_epsdec = results['e-greedy-decay']
    r_eps, t_eps = results['e-greedy']
    r_random, t_random = results['random']

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