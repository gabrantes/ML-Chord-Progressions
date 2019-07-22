"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/21/2019
Filename: train.py
Description: 
    Trains the RandomForest.
"""

from model.data_generator import read_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.chorus.satb import Satb
from utils.utils import num_to_note

import numpy as np
import pandas as pd

def train():

    train_X, train_S, train_A, train_T, train_B = read_data("./data/train.txt")
    train_Y = np.hstack((train_S, train_A, train_T, train_B))

    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("Train inputs and outputs must have similar shape.", train_X.shape, train_Y.shape)

    val_X, val_S, val_A, val_T, val_B = read_data("./data/val.txt")
    val_Y = np.hstack((val_S, val_A, val_T, val_B))

    test_X, test_S, test_A, test_T, test_B = read_data("./data/val.txt")
    test_Y = np.hstack((test_S, test_A, test_T, test_B))

    if test_X.shape[0] != test_Y.shape[0]:
        raise ValueError("Test inputs and outputs must have similar shape.", test_X.shape, test_Y.shape)
    

    # clf = MultiOutputClassifier(
    #     RandomForestClassifier(n_estimators=100)
    # )
    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(train_X, train_Y)

    """
    PREDICTION
    """
    # get predictions
    pred_Y = clf.predict(test_X)

    # get accuracy
    f1 = [0] * 4
    precision = [0] * 4
    recall = [0] * 4
    pred_val_Y = clf.predict(val_X)
    for i in range(4):
        f1[i] = f1_score(val_Y[:, i], pred_val_Y[:, i], average='micro')
        precision[i] = precision_score(val_Y[:, i], pred_val_Y[:, i], average='micro')
        recall[i] = recall_score(val_Y[:, i], pred_val_Y[:, i], average='micro')

    # get gt key
    key_num = test_X[:, :12]
    key = np.zeros((test_X.shape[0], 2), dtype=np.uint8)
    key[:, 0] = np.argmax(key_num, axis=1)
    key[:, 1] = test_X[:, 12]

    # get gt cur chord
    test_cur = test_X[:, 13:17]

    # get gt next chord info
    next_deg = np.argmax(test_X[:, 17:24], axis=1) + 1
    next_sev = test_X[:, 24].astype('uint8')
    next_inv = np.argmax(test_X[:, 25:29], axis=1)

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


    df = pd.DataFrame(columns=[
        'key', 'maj/min', 'cur_chord',
        'next_deg', 'next_sev', 'next_inv',
        'pred_next', 'gt_next', 'notes_correct'
        ])

    df['key'] = key_note
    df['maj/min'] = key[:, 1]

    df['cur_chord'] = test_cur

    df['next_deg'] = next_deg
    df['next_sev'] = next_sev
    df['next_inv'] = next_inv

    df['pred_next'] = pred_out
    df['gt_next'] = test_out
    df['notes_correct'] = notes_correct

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