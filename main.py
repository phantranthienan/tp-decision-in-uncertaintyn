from bandit_run import run_scenario
from plot_results import plot_results

def main():
    K = 9
    N = 1000000
    success_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_runs = 1
    results = run_scenario(K, N, success_rates, num_runs)

    plot_results(results, N)
    
if __name__ == "__main__":
    main()