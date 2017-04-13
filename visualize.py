import numpy as np
import matplotlib.pyplot as plt


def get_rewards(filename, prefix=''):
    f = open(prefix+filename, 'r')

    rewards = []
    for line in f.readlines():
        rewards.append(float(line))

    rewards = rewards[3:]
    return rewards[3:]


def read_all(filename, prefix=''):
    f = open(prefix + filename, 'r')
    rewards = []
    for line in f.readlines():
        if len(line) == 1:
            continue
        l = []
        for s in line.split():
            try:
                l.append(float(s))
            except ValueError:
                l.append(0.0)
        if len(l) < 1000000:
            l += [0.0 for _ in range(len(l), 1000000)]
        rewards.append(l)
    rewards = np.concatenate([rewards], axis=0)
    return rewards


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


def draw_mean(x_list, l_list, N=20000, logscale=True, optimal=None, std=False, title=None):
    for x, l in zip(x_list, l_list):
        xm = np.mean(x, axis=0)
        xm_ = running_mean(xm, N)
        p = plt.plot(xm_, label=l)
        if std:
            xs = []
            for i in range(0, x.shape[0]):
                xs.append(np.reshape(running_mean(x[i], N), [1, -1]))
            xs = np.concatenate(xs, axis=0)
            xs_ = np.std(xs, axis=0)
            c = p[0].get_color()
            y1, y2 = xm_ - xs_, xm_ + xs_
            plt.fill_between(np.arange(0, 1000000-N+1), y1, y2, facecolor=c, alpha=0.2, interpolate=True)
    if optimal:
        plt.plot(np.arange(0, 1000000), np.ones([1000000]) * optimal, label='optimal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.gca().set_xlim([1000, 1000000])
    if title:
        plt.title(title)
    if logscale:
        plt.gca().set_xscale('log')
    else:
        plt.gca().set_xscale('linear')
    plt.show()