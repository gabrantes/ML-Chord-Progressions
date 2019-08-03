"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/29/2019
Filename: stage2_grid.py
Description: 
    Perform grid search to optimize hyperparameters for stage 2.
"""

from sklearn.model_selection import GridSearchCV
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

    # Setup parameters and distributions for Grid Search
    param_dist = {
        'n_estimators': [25, 50],
    }
    scorer = make_scorer(metrics.accuracy_score)

    # train model
    clf = RandomForestClassifier(
        n_estimators=25,
        bootstrap=False,
        criterion='entropy',
        class_weight='balanced',
        max_features=8
    )
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_dist,
        cv=5,
        scoring=scorer,
        verbose=2
    )
    grid_search.fit(df.to_numpy(dtype=np.int8), Y)
    
    search_report(grid_search.cv_results_)

if __name__ == "__main__":    
    parser  = argparse.ArgumentParser(
        description='Perform grid search to optimize hyperparameters for stage 2.'
        )

    search()