import numpy as np
import time
from bandit_env import pull, initialize_bandit

def bandit_random(K, N, success_rates):
    s, n, total_reward = initialize_bandit(K, success_rates)

    for _ in range(K, N):
        chosen_arm = np.random.randint(K)
        
        r = pull(chosen_arm, success_rates)
        s[chosen_arm] += r
        n[chosen_arm] += 1
        total_reward += r

    return total_reward

def bandit_epsilon_greedy(K, N, success_rates, epsilon=0.1):
    s, n, total_reward = initialize_bandit(K, success_rates)

    for _ in range(K, N):
        if np.random.rand() < epsilon:
            chosen_arm = np.random.randint(K)
        else:  
            chosen_arm = np.argmax(s/n)
        
        r = pull(chosen_arm, success_rates)
        s[chosen_arm] += r
        n[chosen_arm] += 1
        total_reward += r

    return total_reward

def bandit_epsilon_greedy_decay(K, N, success_rates):
    s, n, total_reward = initialize_bandit(K, success_rates)

    for t in range(K, N):
        epsilon = 1 / (2 * np.log(t))
        if np.random.rand() < epsilon:
            chosen_arm = np.random.randint(K)
        else:
            chosen_arm = np.argmax(s/n)
        
        r = pull(chosen_arm, success_rates)
        s[chosen_arm] += r
        n[chosen_arm] += 1
        total_reward += r

    return total_reward

def bandit_ucb(K, N, success_rates):   
    s, n, total_reward = initialize_bandit(K, success_rates)

    for t in range(K, N):
        ucb_values = s/n + np.sqrt(2*np.log(t)/n)
        chosen_arm = np.argmax(ucb_values)

        r = pull(chosen_arm, success_rates)
        s[chosen_arm] += r
        n[chosen_arm] += 1
        total_reward += r
    
    return total_reward

def main():
    # Define parameters
    K = 5
    N = 100000
    success_rates = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Run each algorithm & measure final reward and execution time
    algorithms = [
        ("random", bandit_random),
        ("ε-greedy", bandit_epsilon_greedy),
        ("ε-greedy-dec", bandit_epsilon_greedy_decay),
        ("UCB", bandit_ucb)
    ]

    final_rewards = []
    exec_times = []

    for (name, algo) in algorithms:
        start_time = time.time()
        total_reward = algo(K, N, success_rates)
        elapsed = time.time() - start_time

        final_rewards.append(total_reward)
        exec_times.append(elapsed)

        print(f"{name} -> Reward: {total_reward}, Time: {elapsed:.4f} s")

main()