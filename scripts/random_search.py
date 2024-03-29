"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/29/2019
Filename: stage2_random.py
Description: 
    Perform randomized search to optimize hyperparameters for stage 2.
"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

from utils.utils import search_report, get_chord_notes_np
from utils import metrics

import numpy as np
import pandas as pd
import argparse

RANDOM_STATE = 42

def search():

    # read csv into DataFrame
    df = pd.read_csv(
        "./data/chords.csv",
        header=0,
        dtype=np.int8
        )

    # THE ALGORITHM
    next_notes = get_chord_notes_np(df['tonic'], df['maj_min'], df['next_degree'], df['next_seventh'])
    for i in range(12):
        df[str(i)] = next_notes[:, i].tolist()

    # remove outputs
    Y = df[['next_s', 'next_a', 'next_t', 'next_b']].copy()
    Y = Y.to_numpy(dtype=np.int8)
    df.drop(['next_s', 'next_a', 'next_t', 'next_b'], axis=1, inplace=True)

    # feature selection
    # remove unnecessary info
    df.drop(['tonic', 'maj_min'], axis=1, inplace=True)
    df.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    df.drop(['next_degree'], axis=1, inplace=True)

    # Setup parameters and distributions for Randomized Search
    param_dist = {
        'max_features': [2, 4, 6, 8, 10],
        'class_weight': ['balanced', 'balanced_subsample', None],
    }
    scorer = make_scorer(metrics.accuracy_score)

    # train model
    clf = RandomForestClassifier(
        n_estimators=25,
        bootstrap=False,
        criterion='entropy',
    )
    n_iter_search = 40
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=5,
        scoring=scorer,
        random_state=RANDOM_STATE,
        verbose=2
    )
    random_search.fit(df.to_numpy(dtype=np.int8), Y)
    
    search_report(random_search.cv_results_)

if __name__ == "__main__":    
    parser  = argparse.ArgumentParser(
        description='Perform randomized search to optimize hyperparameters for stage 2.'
        )

    search()