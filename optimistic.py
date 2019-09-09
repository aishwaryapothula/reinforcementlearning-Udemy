from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from compare_epsilon import run_experiment_eps

#bandits are your slot machines

class Bandit:
    def __init__(self, m, upper_limit):
        # The true mean of each bandit
        self.m = m
        # The mean calculated for the bandit till now
        self.mean = upper_limit
        # The number of times we pull the bandit
        self.N = 1
    
    def pull(self):
        # Reward for pulling this bandit. Gaussian distribution
        return np.random.randn() + self.m
    
    def update(self, x):
        # update the number of pulls
        self.N += 1
        # update the mean including the new reward x
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
    
def run_experiment_oiv(m1, m2, m3, N, upper_limit = 10):
    # takes in three different means for the three bandits created
    # takes eps value to experiment using different values of epsilon
    # creating 3 different slot machines and store them in bandits.
    # creating objects to class bandit and storing those objects in one variable
    bandits = [(Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit))]
    # data stores mean reward after every pull
    data = np.empty(N)
    
    for i in range(N):
        j = np.argmax([b.mean for b in bandits])
        # store the mean reward including the latest reward for the choosen slot machine in x
        x = bandits[j].pull()
        bandits[j].update(x)
        
        # update data or the mean reward for the latest pull in the data
        data[i] = x

    # calculating the average reward for the experiment
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    
    plt.plot(cumulative_average)
    # average reward calculated at each point is plotted for each of the slot machines
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()
    
    # average reward calculated for each of the slot machines
    for b in bandits:
        print(b.mean)

    return cumulative_average
    
if __name__ == '__main__':
    # bandit 3 has the higest average reward
    # The agent has to figure this out
    # After training the agent must learn to pull bandit 3 most often
    c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 1000000)
    o_v = run_experiment_oiv(1.0, 2.0, 3.0, 100000, upper_limit = 10)
    
    # log scale plot
    plt.plot(c_1, label = 'eps = 0.1')
    plt.plot(o_v, label = 'optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # Linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(o_v, label='optimistic')
    plt.legend()
    plt.show()






