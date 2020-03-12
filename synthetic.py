import numpy as np
import utils
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import multiprocessing as mp

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

def run_exp(intup):
    global X_train, X_test, y_train, y_test
    rep, i, p = intup

    # Make noisy data, simulate pool-based case
    X_train_noisy = utils.add_gaussian_noise(X_train, p)
    y_train_noisy = y_train  # utils.flip_labels(y_train, p)
    X_seed, X_pool = X_train_noisy[:n_seed], X_train_noisy[n_seed:]
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
    return history

np.random.seed(165)
X_train, X_test, y_train, y_test = load_data('wdbc_data.csv', 300)

n_seed = 5
query_budget = 30
reps = 100
ps = np.arange(0, 3.1, 1.0)  # noise level
log_interval = 10
all_results = np.zeros((reps, len(ps), query_budget - n_seed))

rtup = []
for rep in range(reps):
    for i, p in enumerate(ps):
        rtup.append((rep, i, p))

pool = Pool(mp.cpu_count())
histories = pool.map(run_exp, rtup)

for it in range(len(rtup)):
    rep, i, p = rtup[it]
    all_results[rep, i] = histories[it] 


tags = list(map(lambda p: 'var=' + str(p), ps))
results = np.mean(all_results, axis=0)
utils.plot_learning_curves(results, range(n_seed + 1, query_budget + 1),
                           tags, './figures/GN_entropy.png')
