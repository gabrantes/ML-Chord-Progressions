"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/21/2019
Filename: train.py
Description: 
    Trains the RandomForest.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.chorus.satb import Satb
from utils.utils import num_to_note
import utils.metrics as metrics

import numpy as np
import pandas as pd
import time

def train():
    times = {}
    t_start = time.time()

    df = pd.read_csv(
        "./data/chords.csv",
        header=0,
        dtype=np.int8
        )
    
    # print(df['maj_min'].value_counts(normalize=True))
    # print("\n")
    # print(df['next_degree'].value_counts(normalize=True))
    # print("\n")
    # print(df['next_seventh'].value_counts(normalize=True))
    # print("\n")
    # print(df['next_inversion'].value_counts(normalize=True))
    # print("\n")

    # split outputs    
    Y = df[['next_s', 'next_a', 'next_t', 'next_b']].copy()
    df.drop(['next_s', 'next_a', 'next_t', 'next_b'], axis=1, inplace=True)

    # feature selection
    # remove info about cur chord except for voicings
    df.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)

    """
    # consolidate next_seventh and next_inversion to get rid of boolean in next_seventh
    df['next_inv'] = df['next_seventh']*3 + df['next_inversion']
    df.drop(['next_seventh', 'next_inversion'], axis=1, inplace=True)
    """

    """
    # consolidate tonic and maj_min to get rid of boolean in maj_min
    df['key'] = df['maj_min']*12 + df['tonic']
    df.drop(['maj_min', 'tonic'], axis=1, inplace=True)
    """

    # print(df.head())

    columns = df.columns.values.tolist()

    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.2)

    t_data =  time.time()
    times['data'] = t_data - t_start

    # train model
    clf = RandomForestClassifier(
        n_estimators=25,
        random_state=42,
        bootstrap=True,
        max_features=8
        )
    clf.fit(X_train, Y_train)

    feature_importances = pd.DataFrame(
        clf.feature_importances_,
        index = X_train.columns,
        columns=['importance']
        ).sort_values('importance', ascending=False)

    print('\nFeature Importances:\n' + feature_importances.to_string())

    t_train = time.time()
    times['train'] = t_train - t_data
    
    # get predictions
    Y_pred = clf.predict(X_test)
    Y_test = np.asarray(Y_test)

    t_predict = time.time()
    times['predict'] = t_predict - t_train
    
    accuracy = metrics.accuracy_df(Y_test, Y_pred)
    print('\nAccuracy:\n' + accuracy.to_string())

    t_metrics = time.time()
    times['metrics'] = t_metrics - t_predict   

    out_df = pd.DataFrame(
        X_test, 
        columns=columns
        )

    out_df['tonic'] = out_df['tonic'].apply(num_to_note)

    for key in ['cur_s', 'cur_a', 'cur_t', 'cur_b']:
        out_df[key] = out_df[key].apply(num_to_note)
    out_df['cur'] = out_df[['cur_s', 'cur_a', 'cur_t', 'cur_b']].values.tolist()
    out_df.drop(['cur_s', 'cur_a', 'cur_t', 'cur_b'], axis=1, inplace=True)

    gt_next = np.vectorize(num_to_note)(Y_test)
    out_df['gt_next'] = gt_next.tolist()

    pred_next = np.vectorize(num_to_note)(Y_pred)
    out_df['pred_next'] = pred_next.tolist()

    print("\nOutput:\n")
    print(out_df.head())
    out_df.to_csv('output.csv')

    t_output = time.time()
    times['output'] = t_output - t_metrics

    time_str = "\nTotal: {:.3f}s, Data: {:.3f}s, Train: {:.3f}s, Predict: {:.3f}s, Metrics: {:.3f}s, Output: {:.3f}s" \
        .format(t_output - t_start, times['data'], times['train'], times['predict'], times['metrics'], times['output'])
    print(time_str)

if __name__ == "__main__":    
    train()