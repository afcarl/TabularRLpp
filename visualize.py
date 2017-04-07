import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('f', type=str)
args = parser.parse_args()

f = open(args.f, 'r')

rewards = []
for line in f.readlines():
    rewards.append(float(line))

rewards = rewards[3:]


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


results = running_mean(rewards, 50000)
plt.plot(results)
ax = plt.gca()
ax.set_xlim(xmin=1000)
ax.set_xscale('log')
plt.show()
