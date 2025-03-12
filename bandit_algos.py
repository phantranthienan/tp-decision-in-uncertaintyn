import numpy as np
import random

def bandit_random(s, n, t):
    return random.randint(0, len(s)-1)

def bandit_epsilon_greedy(epsilon = 0.1):
    def algo(s, n, t):
        if np.random.rand() < epsilon:
            return random.randint(0, len(s)-1)
        else:
            return np.argmax(s/n)
    return algo

def bandit_epsilon_greedy_decay(s, n, t):
    epsilon = 1/np.log(t*t)
    if np.random.rand() < epsilon:
        return random.randint(0, len(s)-1)
    else:
        return np.argmax(s/n)
    
def bandit_ucb(s, n, t):
    ucb_values = s/n + np.sqrt(2*np.log(t)/n)
    return np.argmax(ucb_values)