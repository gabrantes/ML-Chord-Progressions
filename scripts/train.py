"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/31/2019
Filename: train.py
Description: 
    The task of predicting voice chords is split into 2 parts:

    (1) Getting the notes belonging to the next chord, which is
        handled by an algorithm (get_chord_notes from utils.utils)
    
    (2) Predicting the next chord's voicings using a random forest.
"""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from utils.chorus.satb import Satb
from utils.utils import get_chord_notes_np, num_to_note_np, convert_key
import utils.metrics as metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse

RANDOM_STATE = 42

def train(verbose=1):
    times = {}
    t_start = time.time()
    if verbose > 0:
        print("\nLoading data...")

    # read csv into DataFrame
    df = pd.read_csv(
        "./data/chords.csv",
        header=0,
        dtype=np.int8
        )
    
    if verbose > 1:
        print("\nData Distribution:")
        print(df['maj_min'].value_counts(normalize=True))
        print("\n")
        print(df['next_degree'].value_counts(normalize=True))
        print("\n")
        print(df['next_seventh'].value_counts(normalize=True))
        print("\n")
        print(df['next_inversion'].value_counts(normalize=True))

    # THE ALGORITHM
    next_notes = get_chord_notes_np(df['tonic'], df['maj_min'], df['next_degree'], df['next_seventh'])
    for i in range(12):
        df[str(i)] = next_notes[:, i].tolist()

    # remove outputs
    Y = df[['next_s', 'next_a', 'next_t', 'next_b']].copy()
    df.drop(['next_s', 'next_a', 'next_t', 'next_b'], axis=1, inplace=True)

    # feature selection
    # remove unnecessary info
    extra = df[['tonic', 'maj_min']].copy()
    df.drop(['tonic', 'maj_min'], axis=1, inplace=True)
    df.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    extra[['next_degree', 'next_seventh', 'next_inversion']] = df[['next_degree', 'next_seventh', 'next_inversion']].copy()
    df.drop(['next_degree'], axis=1, inplace=True)

    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.1, random_state=RANDOM_STATE)

    t_data =  time.time()
    times['data'] = t_data - t_start
    if verbose > 0:
        print("\nTraining model...") 

    # train model
    # (hyperparameters optimized by iterative randomized search and grid search as of 7/30/2019)
    clf = RandomForestClassifier(
        n_estimators=270,        
        criterion='gini',
        bootstrap=False,
        max_features=2,
        class_weight='balanced',
        random_state=RANDOM_STATE
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
    if verbose > 0:
        print("\nGenerating predictions...")
    
    # get predictions
    Y_pred = clf.predict(X_test)
    Y_test = np.asarray(Y_test)

    t_predict = time.time()
    times['predict'] = t_predict - t_train
    if verbose > 0:
        print("\nScoring accuracy...")
    
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
        total_hist = plt.hist(total_acc, bins='auto', label='Total')
        plt.title("Accuracy Histogram")
        plt.xlabel("Accuracy")
        plt.ylabel("Count")
        for i in range(len(total_hist[0])):
            if total_hist[0][i] >= 25:
                plt.text(total_hist[1][i], total_hist[0][i], "{:.0f}".format(total_hist[0][i]))
        plt.show(block=False)

    t_metrics = time.time()
    times['metrics'] = t_metrics - t_predict   

    # transform raw model output and format into DataFrame
    out_df = pd.DataFrame(
        X_test, 
        columns=X_test.columns
        )

    for val in extra.columns.values.tolist():
        out_df[val] = extra[val]    

    tonic_np = out_df['tonic'].to_numpy()
    maj_min_np = out_df['maj_min'].to_numpy()

    # current chord voicings: int -> note
    cur_chords = out_df[['cur_s', 'cur_a', 'cur_t', 'cur_b']].to_numpy()
    out_df['cur'] = list(num_to_note_np(tonic_np, maj_min_np, cur_chords))
    out_df.drop(['cur_s', 'cur_a', 'cur_t', 'cur_b'], axis=1, inplace=True)

    # ground truth next chord: int -> note
    out_df['gt_next'] = list(num_to_note_np(tonic_np, maj_min_np, Y_test))

    # predicted next chord: int -> note
    out_df['pred_next'] = list(num_to_note_np(tonic_np, maj_min_np, Y_pred))

    # convert key signature
    out_df['key'] = list(convert_key(tonic_np, maj_min_np))
    out_df.drop(['tonic', 'maj_min'], axis=1, inplace=True)

    out_df['total_accuracy'] = total_acc

    # rearrange / reorganize columns
    out_df = out_df[[
        'key', 'cur', 
        'next_degree', 'next_seventh', 'next_inversion',
        'gt_next', 'pred_next', 'total_accuracy']]

    if verbose > 0:
        print("\nOutput:")
        print(out_df.head())
    out_df.to_csv('./output/output.csv')

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
        plt.show()

if __name__ == "__main__":    
    parser  = argparse.ArgumentParser(
        description='Train and evaluate the random forest classifier for predicting chord voicings.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-v", "--verbose",
        help="0: silent \
            \n1: accuracy, output \
            \n2: dataset distribution, feature importances, accuracy, output \
            \nDEFAULT: 1",
        default=1)
    args = parser.parse_args()

    train(verbose=int(args.verbose))