import numpy as np
import utils
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
from modAL.uncertainty import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import multiprocessing as mp
import pickle as pkl
import random

def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return [query_idx], X_pool[query_idx]

def load_music(ftrain, ftest, p_keep):
    """Load a dataset CSV, shuffle, and split into train and test."""
    data_train = pkl.load(open(ftrain, 'rb'))
    data_test = pkl.load(open(ftest, 'rb'))

    np.random.shuffle(data_train)

    X_train = []
    for i in data_train:
        X_train.append(i[0])
    X_train = np.array(X_train)

    tset = set([])
    for i in data_train:
        for l in i[1]:
            tset.add(l)

    tset = list(tset)
    total_j = len(tset)
    tmap = {tset[i]:i for i in range(len(tset))}
    
    y_train = []
    for i in data_train:
        tmp = [tmap[x] for x in i[1]]
        gt = np.bincount(tmp).argmax()

        if random.random() < p_keep:
            #tsamprnd = []
            #for j in i[1]:
            #    if tmap[j] != gt:
            #        tsamprnd.append(tmap[j])
            #if(len(tsamprnd) == 0):
            #    y_train.append(gt)
            #else:
            y_train.append(tmap[random.choice(i[1])])
        else:
            y_train.append(gt)
    y_train = np.array(y_train)
    y_gt = []
    for i in data_train:
        tmp = [tmap[x] for x in i[1]]
        y_gt.append(np.bincount(tmp).argmax())
    y_gt = np.array(y_gt)
    
    X_test = []
    for i in data_test:
        X_test.append(i[0])
    X_test = np.array(X_test)

    y_test = []
    for i in data_test:
        y_test.append(tmap[i[1][0]])
    y_test = np.array(y_test)

    scaler = StandardScaler()    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    tmp = [1 if y_train[i] != y_gt[i] else 0 for i in range(len(y_train))]
    
    return X_train, X_test, y_train, y_test, sum(tmp)/len(tmp)

def run_exp_music(intup):
    global X_train, X_test, y_train, y_test
    rep, strat, n_seed = intup

    seed = np.random.choice(range(len(X_train)), n_seed, replace=False)
    other = list(set(range(len(X_train))) - set(seed))
    
    X_seed, X_pool = X_train[seed], X_train[other]
    y_seed, y_pool = y_train[seed], y_train[other]

    # Initializing the learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(n_estimators=60, max_depth=30),
        query_strategy=strat,
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
dataset = 'music'

n_seed = 10
query_budget = 150
reps = 1000
log_interval = 10
all_results = np.zeros((reps, query_budget - n_seed))

tags = []
random_results = []

ps = [0, 0.25, 0.5, 0.75, 1.0]
labels = []
fig, axs = plt.subplots(1, len(ps), figsize=(18, 4))
xs = [_ for _ in range(n_seed+1, query_budget + 1)]

for i in range(len(ps)):
    p = ps[i]
    X_train, X_test, y_train, y_test, kappa = load_music('music_train.pkl', 'music_test.pkl', p)
    print(kappa)
    axs[i].set_title("p {:.02f}, disagreement {:.02f}".format(p, kappa))

    rtup = []
    for rep in range(reps):
        rtup.append((rep, random_sampling, n_seed))
    pool = Pool(mp.cpu_count())
    results = np.array(pool.map(run_exp_music, rtup))
    mn = np.mean(results, axis=0)
    sd = np.std(results, axis=0)
    print(results.shape, mn.shape, sd.shape)
    axs[i].plot(xs, mn, color='orange', label='random')
    axs[i].fill_between(xs, mn - sd, mn + sd, facecolor='orange', alpha=0.2)
    pool.close()
    
    rtup = []
    for rep in range(reps):
        rtup.append((rep, uncertainty_sampling, n_seed))
    pool = Pool(mp.cpu_count())
    results = np.array(pool.map(run_exp_music, rtup))
    mn = np.mean(results, axis=0)
    sd = np.std(results, axis=0)
    axs[i].plot(xs, mn, color='blue', label='uncertainty')
    axs[i].fill_between(xs, mn - sd, mn + sd, facecolor='blue', alpha=0.2)
    axs[i].set_ylim(0, 0.5)
    axs[i].grid()
    
fig.text(0.5, 0.02, '# of labels queried', ha='center')
fig.text(0.08, 0.5, 'test accuracy', va='center', rotation='vertical')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='right')

plt.savefig('test.png')

