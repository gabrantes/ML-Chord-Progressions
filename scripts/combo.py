"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/29/2019
Filename: combo.py
Description: 
    Links the Stage 1 Model and the Stage 2 Model for joint inferrence/evaluation.
"""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from utils.chorus.satb import Satb
from utils.utils import num_to_note
import utils.metrics as metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    # remove outputs
    Y = df[['next_s', 'next_a', 'next_t', 'next_b']].copy()
    df.drop(['next_s', 'next_a', 'next_t', 'next_b'], axis=1, inplace=True)

    # train/test split
    A_feat, B_feat, A_gt, B_gt = train_test_split(df, Y, test_size=0.5, random_state=RANDOM_STATE)

    # converting back to DataFrame
    A_feat = pd.DataFrame(
       data=A_feat,
       columns=df.columns,
       dtype=np.int8 
    )

    B_feat = pd.DataFrame(
        data=B_feat,
        columns=df.columns,
        dtype=np.int8
    )

    # preparing stage 1 data
    stage1_gt = pd.DataFrame(
        data=A_gt,
        columns=['next_s', 'next_a', 'next_t', 'next_b'],
        dtype=np.int8
    )

    mlb = MultiLabelBinarizer(
        classes=list(range(0, 12))
    )

    stage1_gt = mlb.fit_transform(
        stage1_gt.applymap(
            lambda x: x % 12
        ).values.tolist()
    )

    stage1_train = A_feat.copy()    
    stage1_train.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    stage1_train.drop(['cur_s', 'cur_a', 'cur_t','cur_b'], axis=1, inplace=True)
    stage1_train.drop(['next_inversion'], axis=1, inplace=True)

    stage1_test = B_feat.copy()    
    stage1_test.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    stage1_test.drop(['cur_s', 'cur_a', 'cur_t','cur_b'], axis=1, inplace=True)
    stage1_test.drop(['next_inversion'], axis=1, inplace=True)
    # finished preparing stage 1 data

    # train stage 1 model
    print("\nTraining stage 1...")
    stage1_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        bootstrap=False,
        class_weight='balanced'
        )
    stage1_model.fit(stage1_train, stage1_gt)

    # get stage 1 predictions
    print("Predicting on stage 1...")
    stage1_pred = stage1_model.predict(stage1_test)

    # preparing stage 2 data
    stage2_train = B_feat.copy()
    stage2_train.drop(['tonic', 'maj_min'], axis=1, inplace=True)
    stage2_train.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    stage2_train.drop(['next_degree'], axis=1, inplace=True)

    stage2_test = A_feat.copy()
    stage2_test.drop(['tonic', 'maj_min'], axis=1, inplace=True)
    stage2_test.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    stage2_test.drop(['next_degree'], axis=1, inplace=True)

    for i in range(12):
        stage2_train[str(i)] = stage1_pred[:, i].tolist()
        stage2_test[str(i)] = stage1_gt[:, i].tolist()
    # finished preparing stage 2 data

    # train stage 2 model
    print("\nTraining stage 2...")
    stage2_model = RandomForestClassifier(
        n_estimators=270,
        random_state=RANDOM_STATE,
        bootstrap=False,
        max_features=2,
        class_weight='balanced'
        )
    stage2_model.fit(stage2_train, B_gt)
    
    # get predictions
    print("Predicting on stage 2...")
    Y_pred = stage2_model.predict(stage2_test)
    Y_test = np.asarray(A_gt)
    
    # score accuracy
    print("\nEvaluating metrics...")
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
        hist = plt.hist(total_acc, bins='auto')
        plt.title("Multistage: Accuracy Histogram")
        plt.xlabel("Accuracy")
        plt.ylabel("Count")
        for i in range(len(hist[0])):
            plt.text(hist[1][i], hist[0][i], str(int(hist[0][i])))
        #plt.show(block=False)
        plt.show()

    exit()
    """
    LEFT OFF HERE
    """

    # transform raw model output and format into DataFrame
    out_df = pd.DataFrame(
        X_test, 
        columns=X_test.columns
        )

    # out_df['tonic'] = out_df['tonic'].apply(num_to_note)

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
        'cur', 
        'gt_next', 'pred_next', 'total_accuracy']]

    if verbose > 0:
        print("\nOutput:")
        print(out_df.head())
    out_df.to_csv('output.csv') 

    if verbose > 0:
        plt.show()

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