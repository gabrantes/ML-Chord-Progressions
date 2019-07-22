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

import numpy as np
import pandas as pd

def train():

    df = pd.read_csv(
        "./data/chords.csv",
        header=0
        )

    # split outputs    
    Y = df[['next_s', 'next_a', 'next_t', 'next_b']].copy()
    df.drop(['next_s', 'next_a', 'next_t', 'next_b'], axis=1, inplace=True)

    # feature selection (remove info about cur chord except for voicings)
    df.drop(['cur_degree', 'cur_seventh', 'cur_inversion'], axis=1, inplace=True)
    columns = df.columns.values.tolist()

    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, Y_train)

    """
    PREDICTION
    """
    # get predictions
    Y_pred = clf.predict(X_test)
    Y_out = np.vectorize(num_to_note)(Y_pred)

    out_df = pd.DataFrame(
        X_test, 
        columns=columns
        )

    out_df['cur_s'] = out_df['cur_s'].apply(num_to_note)
    out_df['cur_a'] = out_df['cur_a'].apply(num_to_note)
    out_df['cur_t'] = out_df['cur_t'].apply(num_to_note)
    out_df['cur_b'] = out_df['cur_b'].apply(num_to_note)
        
    Y_test = Y_test.applymap(num_to_note)
    out_df = pd.concat([out_df, Y_test], axis=1)
    out_df['next_pred'] = Y_out.tolist()
    print(out_df.head())
    exit()
    """
    LEFT OFF HERE

    1. convert cur chord to 1 column where each element is a list
    2. same with next gt
    3. create flags for easy scaling of chords
    4. same for one hot encoding
    5. fix accuracy
    6. notes correct and num correct
    """

    # get accuracy
    """
    f1 = [0] * 4
    precision = [0] * 4
    recall = [0] * 4
    for i in range(4):
        f1[i] = f1_score(Y_test[:, i], Y_pred[:, i], average='micro')
        precision[i] = precision_score(Y_test[:, i], Y_pred[:, i], average='micro')
        recall[i] = recall_score(Y_test[:, i], Y_pred[:, i], average='micro')
    """

    # convert all ints to notes
    satb = Satb()

    pred_out = [None] * pred_Y.shape[0]
    for i in range(pred_Y.shape[0]):
        try:
            pred_out[i] = [num_to_note(el) for el in satb.unscale(pred_Y[i, :])]
        except Exception as e:
            pred_out[i] = ["INVALID"]

    # pred_Y = np.apply_along_axis(satb.unscale, 1, pred_Y)
    test_Y = np.apply_along_axis(satb.unscale, 1, test_Y)
    test_cur = np.apply_along_axis(satb.unscale, 1, test_cur)

    np_to_note = np.vectorize(num_to_note)

    # pred_out = np_to_note(pred_Y).tolist()
    test_out = np_to_note(test_Y).tolist()
    test_cur = np_to_note(test_cur).tolist()

    key_note = np_to_note(key[:, 0]).tolist()

    num_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    notes_correct = [0] * len(test_out)
    for i in range(len(test_out)):
        count = 0
        for j in range(4):
            if len(pred_out[i]) != 4:
                break
            if test_out[i][j] == pred_out[i][j]:
                count += 1
        num_correct[count] += 1
        notes_correct[i] = count

    print("\n")
    print("Notes correct\t# chords\t% chords")
    for key, val in num_correct.items():
        print("{}\t\t{}\t\t{}".format(key, val, val/len(test_Y)))
            
    print("\nPrecision:\n\t{}".format(precision))
    print("\nRecall:\n\t{}".format(recall))
    print("\nF1:\n\t{}".format(f1))

    df.to_csv('output.csv')

if __name__ == "__main__":
    train()