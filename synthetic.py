import numpy as np
import utils
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_data(filename, n_train):
    """Load a dataset CSV, shuffle, and split into train and test."""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    np.random.seed(165)
    X_train, X_test, y_train, y_test = load_data('wdbc_data.csv', 300)

    n_seed = 5
    query_budget = 50
    reps = 100
    ps = [0, 0.1, 0.2, 0.3]
    log_interval = 10
    all_results = np.zeros((reps, len(ps), query_budget - n_seed))

    for rep in range(reps):
        for i, p in enumerate(ps):
            # Make noisy data, simulate pool-based case
            y_train_noisy = utils.flip_labels(y_train, p)
            X_seed, X_pool = X_train[:n_seed], X_train[n_seed:]
            y_seed, y_pool = y_train_noisy[:n_seed], y_train_noisy[n_seed:]

            # Initializing the learner
            learner = ActiveLearner(
                estimator=RandomForestClassifier(n_estimators=10),
                query_strategy=entropy_sampling,
                X_training=X_seed, y_training=y_seed
            )

            # Run active learning and record history of test accuracy
            history = np.zeros(query_budget - n_seed)
            for j in range(query_budget - n_seed):
                query_idx, query_inst = learner.query(X_pool)
                learner.teach(X_pool[query_idx], y_pool[query_idx])
                history[j] = learner.score(X_test, y_test)
            all_results[rep, i] = history

        if rep % log_interval == log_interval - 1:
            print('rep', rep + 1)

    tags = list(map(lambda p: 'p=' + str(p), ps))
    results = np.mean(all_results, axis=0)
    utils.plot_learning_curves(results, range(n_seed + 1, query_budget + 1),
                               tags, 'synthetic_noisy_labels.png')
