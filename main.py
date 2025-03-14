from bandit_run import run_scenario
from plot_results import plot_results

def main():
    N = 1000000
    num_runs = 10
    
    # First Scenario: A Big Gap Between Arms
    K = 5
    success_rates = [0.1, 0.2, 0.25, 0.3, 0.8]
    results1 = run_scenario(K, N, success_rates, num_runs)

    # Second Scenario: All Arms Are Very Similar
    K = 4
    success_rates = [0.45, 0.46, 0.48, 0.5]
    results2 = run_scenario(K, N, success_rates, num_runs)

    plot_results(results1, N)
    plot_results(results2, N)

    # K = 5
    # success_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    # results = run_scenario(K, N, success_rates, num_runs)
    # plot_results(results, N)

if __name__ == "__main__":
    main()
