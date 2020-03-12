import numpy as np
import utils
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import multiprocessing as mp
import pickle as pkl
import random

def load_data(filename, p_train):
    """Load a dataset CSV, shuffle, and split into train and test."""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    n_train = int(p_train*len(data))
    
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test

def load_music(ftrain, ftest):
    """Load a dataset CSV, shuffle, and split into train and test."""
    data_train = pkl.load(open(ftrain, 'rb'))
    data_test = pkl.load(open(ftest, 'rb'))

    np.random.shuffle(data_train)

    X_train = []
    for i in data_train:
        X_train.append(i[0])
    X_train = np.array(X_train)

    y_train = []
    for i in data_train:
        counts = {}
        for l in i[1]:
            if l in counts:
                counts[l] += 1
            else:
                counts[l] = 1
        label_max = None
        lmax = -1
        for key in counts:
            if counts[key] > lmax:
                lmax = counts[key]
                label_max = key
            
        y_train.append(label_max)
    y_train = np.array(y_train)

    X_test = []
    for i in data_test:
        X_test.append(i[0])
    X_test = np.array(X_test)

    y_test = []
    for i in data_test:
        y_test.append(i[1][0])
    y_test = np.array(y_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

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

def run_exp_music(intup):
    global X_train, X_test, y_train, y_test
    rep, i, p = intup

    X_seed, X_pool = X_train[:n_seed], X_train[n_seed:]
    y_seed, y_pool = y_train[:n_seed], y_train[n_seed:]

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
dataset = 'blood.csv'

#X_train, X_test, y_train, y_test = load_data(dataset, 0.8)
X_train, X_test, y_train, y_test = load_music('music_train.pkl', 'music_test.pkl')

n_seed = 15
query_budget = 700
reps = 100
ps = [0]
log_interval = 10
all_results = np.zeros((reps, len(ps), query_budget - n_seed))

rtup = []
for rep in range(reps):
    for i, p in enumerate(ps):
        rtup.append((rep, i, p))

pool = Pool(mp.cpu_count())
histories = pool.map(run_exp_music, rtup)

for it in range(len(rtup)):
    rep, i, p = rtup[it]
    all_results[rep, i] = histories[it] 


tags = list(map(lambda p: 'var=' + str(p), ps))
results = np.mean(all_results, axis=0)
utils.plot_learning_curves(results, range(n_seed + 1, query_budget + 1),
                           tags, '{}_synthetic_noisy_labels.png'.format(dataset.split('.')[0]))

