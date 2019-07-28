"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/21/2019
Filename: stage1.py
Description: 
    Multi-label classification: determine notes in next chord
"""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils.chorus.satb import Satb
from utils.utils import num_to_note
import utils.metrics as metrics

import numpy as np
import pandas as pd
import time
import argparse

RANDOM_STATE = 42

def train(verbose=False):
    times = {}
    t_start = time.time()

    # read csv into DataFrame
    df = pd.read_csv(
        "./data/chords.csv",
        header=0,
        dtype=np.int8
        )
    
    if verbose > 1:
        print(df['maj_min'].value_counts(normalize=True))
        print("\n")
        print(df['next_degree'].value_counts(normalize=True))
        print("\n")
        print(df['next_seventh'].value_counts(normalize=True))
        print("\n")
        print(df['next_inversion'].value_counts(normalize=True))
        print("\n")

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

    print(df.head())

    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.1, random_state=RANDOM_STATE)

    t_data =  time.time()
    times['data'] = t_data - t_start

    # train model
    clf = RandomForestClassifier(
        n_estimators=256,
        random_state=RANDOM_STATE,
        bootstrap=True,
        max_features=4,
        class_weight='balanced'
        )
    clf.fit(X_train, Y_train)

    feature_importances = pd.DataFrame(
        clf.feature_importances_,
        index = X_train.columns,
        columns=['importance']
        ).sort_values('importance', ascending=False)

    if verbose > 1:
        print('\nFeature Importances:\n' + feature_importances.to_string())

    t_train = time.time()
    times['train'] = t_train - t_data
    
    # get predictions
    Y_pred = clf.predict(X_test)
    Y_test = np.asarray(Y_test)

    t_predict = time.time()
    times['predict'] = t_predict - t_train

    for i in range(5):
        print("\ngt:\t{}".format(Y_test[i]))
        print("pred:\t{}".format(Y_pred[i]))
    
    new_acc = accuracy_score(Y_test, Y_pred)
    print("\nAccuracy score:\t{}".format(new_acc))

    exit()
    """
    
    # score accuracy
    total_acc, notes_acc, inversion_acc, voicing_acc = metrics.accuracy_np(Y_test, Y_pred)
    accuracy_df = pd.DataFrame({
        'total_accuracy': total_acc,
        'notes': notes_acc,
        'inversion': inversion_acc,
        'voicing': voicing_acc
    })
    if verbose > 0:
        print("\nMetrics:")
        print(accuracy_df.describe().loc[['mean', 'std', '25%', '50%', '75%']])

    t_metrics = time.time()
    times['metrics'] = t_metrics - t_predict   

    # transform raw model output and format into DataFrame
    out_df = pd.DataFrame(
        X_test, 
        columns=X_test.columns
        )

    out_df['tonic'] = out_df['tonic'].apply(num_to_note)

    # current chord voicings: int -> note
    for key in ['cur_s', 'cur_a', 'cur_t', 'cur_b']:
        out_df[key] = out_df[key].apply(num_to_note)
    out_df['cur'] = out_df[['cur_s', 'cur_a', 'cur_t', 'cur_b']].values.tolist()
    out_df.drop(['cur_s', 'cur_a', 'cur_t', 'cur_b'], axis=1, inplace=True)

    # ground truth next chord: int -> note
    gt_next = np.vectorize(num_to_note)(Y_test)
    out_df['gt_next'] = gt_next.tolist()

    # predicted next chord: int -> note
    pred_next = np.vectorize(num_to_note)(Y_pred)
    out_df['pred_next'] = pred_next.tolist()

    out_df['total_accuracy'] = total_acc

    # rearrange / reorganize columns
    out_df = out_df[[
        'tonic', 'maj_min', 'cur', 
        'next_degree', 'next_seventh', 'next_inversion',
        'gt_next', 'pred_next', 'total_accuracy']]

    if verbose > 0:
        print("\nOutput:")
        print(out_df.head())
    out_df.to_csv('output.csv')

    t_output = time.time()
    times['output'] = t_output - t_metrics   

    if verbose > 0:
        time_str = "\nTotal: {:.3f}s".format(t_output-t_start)
        time_str += ", Data: {:.3f}s".format(times['data'])
        time_str += ", Train: {:.3f}s".format(times['train'])
        time_str += ", Predict: {:.3f}s".format(times['predict'])
        time_str += ", Metrics: {:.3f}s".format(times['metrics'])
        time_str += ", Output: {:.3f}s".format(times['output'])
        print(time_str)
    
    """

if __name__ == "__main__":    
    parser  = argparse.ArgumentParser(
        description='Train and validate random forest classifier.'
        )
    parser.add_argument("-v", "--verbose",
        help="0: silent | 1: accuracy, output | \
            2: dataset distribution, feature importances, accuracy, output | DEFAULT: 1",
        default=1)
    args = parser.parse_args()

    train(verbose=int(args.verbose))