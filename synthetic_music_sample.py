import numpy as np
import utils
from modAL.models import ActiveLearner
from modAL.uncertainty import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import multiprocessing as mp
import pickle as pkl
import random

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
    rep = intup

    X_seed, X_pool = X_train[:n_seed], X_train[n_seed:]
    y_seed, y_pool = y_train[:n_seed], y_train[n_seed:]

    # Initializing the learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(n_estimators=150, max_depth=80),
        query_strategy=uncertainty_sampling,
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

n_seed = 15
query_budget = 150
reps = 50
log_interval = 10
all_results = np.zeros((reps, query_budget - n_seed))

tags = []
full_results = []

for p in [0.1, 0.3, 0.6, 0.8, 1.0]:
    X_train, X_test, y_train, y_test, kappa = load_music('music_train.pkl', 'music_test.pkl', p)
    print(kappa)
    tags.append("disagree {:.02f}".format(kappa))
    rtup = []
    for rep in range(reps):
        rtup.append(rep)
    pool = Pool(mp.cpu_count())
    histories = pool.map(run_exp_music, rtup)
    all_results = np.array(histories)
    results = np.mean(all_results, axis=0)
    full_results.append(results)
    
utils.plot_learning_curves(full_results, range(n_seed + 1, query_budget + 1),
                           tags, '{}_realistic_noisy_labels.png'.format(dataset.split('.')[0]))
