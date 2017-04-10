import numpy as np
import matplotlib.pyplot as plt


def get_rewards(filename, prefix=''):
    f = open(prefix+filename, 'r')

    rewards = []
    for line in f.readlines():
        rewards.append(float(line))

    rewards = rewards[3:]
    return rewards[3:]


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def draw(x_list, l_list, N=20000, logscale=True):
    for x, l in zip(x_list, l_list):
        x_ = running_mean(x, N)
        plt.plot(x_, label=l)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.gca().set_xlim([1000, 1000000])
    if logscale:
        plt.gca().set_xscale('log')
    else:
        plt.gca().set_xscale('linear')
    plt.show()