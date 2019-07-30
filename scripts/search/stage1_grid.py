"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/29/2019
Filename: stage1_grid.py
Description: 
    Perform grid search to optimize hyperparameters for stage 1.
"""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils.utils import search_report

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

    mlb = MultiLabelBinarizer(
        classes=list(range(0, 12))
    )

    Y = mlb.fit_transform(
        df[['next_s', 'next_a', 'next_t', 'next_b']].applymap(
            lambda x: x % 12
        ).values.tolist()
    )

    # remove outputs
    df.drop(['next_s', 'next_a', 'next_t', 'next_b'], axis=1, inplace=True)

    # feature selection
    # remove all info about current chord except for voicings
    df.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    df.drop(['cur_s', 'cur_a', 'cur_t','cur_b'], axis=1, inplace=True)
    df.drop(['next_inversion'], axis=1, inplace=True)
    
    # Setup parameters and distributions for Grid Search
    param_dist = {
        'max_depth': [3, 5, 7, 9, 11, 13, 15, None]
    }

    # train model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_features=2,
        criterion='gini',
        bootstrap=False,
        class_weight='balanced'
    )
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_dist,
        cv=5,
        verbose=2
    )
    grid_search.fit(df.to_numpy(dtype=np.int8), Y)

    search_report(grid_search.cv_results_)

if __name__ == "__main__":    
    parser  = argparse.ArgumentParser(
        description='Perform grid search to optimize hyperparameters for stage 1.'
        )
    
    search()