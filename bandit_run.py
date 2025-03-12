import time

from bandit_env import pull, initialize_bandit

def run_bandit(K, N, success_rates, algo):
    s, n, total_reward = initialize_bandit(K, success_rates)

    for t in range(K, N):
        chosen_arm = algo(s, n, t)

        r = pull(chosen_arm, success_rates)
        s[chosen_arm] += r
        n[chosen_arm] += 1
        total_reward += r
    
    return total_reward

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