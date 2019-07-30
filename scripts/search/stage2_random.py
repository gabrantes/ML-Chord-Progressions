"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/21/2019
Filename: stage2.py
Description: 
    Perform randomized/grid search to optimize hyperparameters for stage 2.
"""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

from utils.chorus.satb import Satb
from utils.utils import num_to_note, search_report
import utils.metrics as metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse

RANDOM_STATE = 42

def search():
    times = {}
    t_start = time.time()

    # read csv into DataFrame
    df = pd.read_csv(
        "./data/chords.csv",
        header=0,
        dtype=np.int8
        )

    mlb = MultiLabelBinarizer(
        classes=list(range(0, 12))
    )

    next_notes = mlb.fit_transform(
        df[['next_s', 'next_a', 'next_t', 'next_b']].applymap(
            lambda x: x % 12
        ).values.tolist()
    )
    for i in range(12):
        df[str(i)] = next_notes[:, i].tolist()

    # remove outputs
    Y = df[['next_s', 'next_a', 'next_t', 'next_b']].copy()
    Y = Y.to_numpy(dtype=np.int8)
    df.drop(['next_s', 'next_a', 'next_t', 'next_b'], axis=1, inplace=True)

    # feature selection
    # remove unnecessary info
    extra = df[['tonic', 'maj_min']].copy()
    df.drop(['tonic', 'maj_min'], axis=1, inplace=True)
    df.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    extra[['next_degree', 'next_seventh', 'next_inversion']] = df[['next_degree', 'next_seventh', 'next_inversion']].copy()
    df.drop(['next_degree', 'next_seventh', 'next_inversion'], axis=1, inplace=True)    

    t_data =  time.time()
    times['data'] = t_data - t_start

    # Setup parameters and distributions for Randomized Search
    param_dist = {
        "n_estimators": [50, 100, 150, 200, 250, 300],
        "max_features": [2, 4, 6, 8, 10, 12, 14, 16],
        "bootstrap": [True, False],
    }
    scorer = make_scorer(metrics.accuracy_score)

    # train model
    clf = RandomForestClassifier()
    n_iter_search = 20
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=5,
        scoring=scorer,
        random_state=RANDOM_STATE,
        verbose=1
    )
    random_search.fit(df.to_numpy(dtype=np.int8), Y)
    
    search_report(random_search.cv_results_)

if __name__ == "__main__":    
    parser  = argparse.ArgumentParser(
        description='Perform randomized/grid search to optimize hyperparameters for stage 2.'
        )

    search()