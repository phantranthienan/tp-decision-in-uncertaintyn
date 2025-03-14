# Multi-Armed Bandit Algorithms Implementation

This project implements and compares different multi-armed bandit algorithms for reinforcement learning.

## Overview

The multi-armed bandit problem is a classic reinforcement learning scenario where an agent must decide between multiple actions (arms) with uncertain rewards to maximize cumulative rewards over time. This project implements several algorithms for solving this problem:

- Random selection
- Epsilon-greedy with fixed epsilon
- Epsilon-greedy with decay
- Upper Confidence Bound (UCB)

## File Structure

- `bandit_setup.py`: Contains utility functions for initializing bandits and pulling arms
- `bandit_algos.py`: Implements the core algorithms for arm selection strategies
- `bandit_run.py`: Provides functions for running experiments and scenarios
- `plot_results.py`: Visualizes the results of experiments
- `main.py`: Main entry point that sets up scenarios and runs experiments
- `test.py`: Contains alternative implementation for testing purposes

## Algorithms

1. **Random**: Selects arms uniformly at random
2. **Epsilon-greedy**: Selects the best arm with probability 1-ε, and a random arm with probability ε
3. **Epsilon-greedy with decay**: Similar to epsilon-greedy, but decreases ε over time
4. **UCB (Upper Confidence Bound)**: Selects arms based on their estimated value plus an exploration bonus

## How to Run

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib

### Running the Main Experiment

```bash
python main.py