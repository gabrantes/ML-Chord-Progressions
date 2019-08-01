"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/21/2019
Filename: stage1.py
Description: 
    Stage 1: Determine notes in next chord (multi-label classification)
"""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils.chorus.satb import Satb
from utils.utils import num_to_note, get_chord_notes
import utils.metrics as metrics

import numpy as np
import pandas as pd
import time
import argparse

RANDOM_STATE = 42

def train(verbose=1):

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

    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.1, random_state=RANDOM_STATE)
    print(X_test.head())
    X_test = X_test.to_numpy(dtype=np.int8)

    if verbose > 0:
        print("\nGenerating predictions...")
    
    # get predictions
    Y_pred = np.zeros((X_test.shape[0], 12), dtype=np.int8)
    for i in range(X_test.shape[0]):
        Y_pred[i, :] = get_chord_notes(X_test[i, 0], X_test[i, 1], X_test[i, 2], X_test[i, 3])
    Y_test = np.asarray(Y_test)
    
    new_acc = accuracy_score(Y_test, Y_pred)   

    count = 0
    for i in range(Y_test.shape[0]):
        if np.any(np.logical_xor(Y_test[i], Y_pred[i])):
            count += 1
            print("")
            print("Tonic: {}, Maj/Min: {}, Degree: {}, Seventh: {}".format(
                X_test[i, 0], X_test[i, 1], X_test[i, 2], X_test[i, 3]
                )
            )
            print("Ground Truth:\t{}".format(Y_test[i]))
            print("Predicted:\t{}".format(Y_pred[i]))
    print("\nTotal count: {} out of {}".format(count, Y_test.shape[0]))
    print("\nPercentage: {}".format(count/Y_test.shape[0]))
    if verbose > 0:
        print("\nAccuracy score:\t{}".format(new_acc))

if __name__ == "__main__":    
    parser  = argparse.ArgumentParser(
        description='Stage 1: Determine notes in next chord (multi-label classification)',
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