import numpy as np
import matplotlib.pyplot as plt

np.random.seed(165)


def flip_labels(Y, p):
    """Returns binary class labels with proportion p randomly flipped."""
    assert set(Y) == {1, 2}
    Y = np.copy(Y)
    for i, e in enumerate(Y):
        if np.random.rand() < p:
            if e == 1:
                Y[i] = 2
            else:
                Y[i] = 1
    return Y


def add_gaussian_noise(X, s):
    """Returns input data with added Gaussian noise of mean 0 and variance s^2."""
    return X + np.random.normal(0, s, X.shape)


def add_uniform_noise(X, s):
    """Returns input data with added uniform noise of mean 0 and variance s^2."""
    return X + np.random.uniform(-s, s, X.shape)


def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return [query_idx], X_pool[query_idx]


def plot_learning_curves(results, xs, leg_tags, outfile, ylabel='accuracy', do_show=False):
    """Plots the given learning curves with the given legend tags and saves to a file."""
    for i, curve in enumerate(results):
        plt.plot(xs, curve, label=leg_tags[i])
    plt.ylim(0.3, 1.0)
    plt.xlabel('# of labels queried')
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    if do_show:
        plt.show()
    plt.savefig(outfile)
