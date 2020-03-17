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
        y_i = []
        which = [_ for _ in range(len(i[1]))]
        random.shuffle(which)
        which = which[0:int(max(1, p_keep*len(which)))]
        for j in which:
            y_i.append(tmap[i[1][j]])
        y_train.append(y_i)

    X_test = []
    for i in data_test:
        X_test.append(i[0])
    X_test = np.array(X_test)

    y_test = []
    for i in data_test:
        y_test.append(tmap[i[1][0]])
    y_test = np.array(y_test)

    y_train_sel = []
    for i in y_train:
        y_train_sel.append(np.bincount(i).argmax())
    y_train_sel = np.array(y_train_sel)
    
    scaler = StandardScaler()
    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train_sel, y_test, fleiss_kappa(y_train, total_j)

def run_exp_music(intup):
    global X_train, X_test, y_train, y_test
    rep = intup

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

def fleiss_kappa(labels, nj):
    pj = [0 for _ in range(nj)]

    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -1:
                pj[labels[i][j]] += 1

    spj = sum(pj)
    pjs = [x/spj for x in pj]

    #print(pjs)

    pis = []
    
    for i in range(len(labels)):
        pi = 0
        ct = 0
        for j in range(len(labels[i])):
            for k in range(j + 1, len(labels[i])):
                ct += 1
                if labels[i][j] == labels[i][k]:
                    pi += 1
        if(ct == 0):
            pis.append(1)
        else:
            pis.append(pi/ct)
    pbar = sum(pis)/len(pis)
    pebar = sum([x**2 for x in pjs])
    return (pbar - pebar)/(1 - pebar)


np.random.seed(165)
dataset = 'music'

n_seed = 15
query_budget = 150
reps = 100
log_interval = 10
all_results = np.zeros((reps, query_budget - n_seed))

tags = []
full_results = []

for p in range(4, 11):
    X_train, X_test, y_train, y_test, kappa = load_music('music_train.pkl', 'music_test.pkl', p/10)
    print(kappa)
    tags.append("kappa {}".format(kappa))
    rtup = []
    for rep in range(reps):
        rtup.append(rep)
    pool = Pool(mp.cpu_count())
    histories = pool.map(run_exp_music, rtup)
    all_results = np.array(histories)
    results = np.mean(all_results, axis=0)
    full_results.append(results)
    
utils.plot_learning_curves(full_results, range(n_seed + 1, query_budget + 1),
                           tags, '{}_synthetic_noisy_labels.png'.format(dataset.split('.')[0]))
