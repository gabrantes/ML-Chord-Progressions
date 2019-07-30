"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/25/2019
Filename: metrics.py
Description: 
    Custom accuracy functions for comparing predicted chordss to ground truth chords.
"""

import numpy as np
import pandas as pd
from utils.chorus.satb import Satb

NOTES_COEF = 0.7
INVERSION_COEF = 0.2
VOICING_COEF = 0.1

def accuracy(gt_next, pred_next):
    """
    Custom accuracy function for comparing predicted chord to ground truth chord.

    Args:
        gt_next: the ground truth chord as [soprano, alto, tenor, bass], where each note is an int
        pred_next: the predicted chord, in the same format as above

    Returns:
        Total accuracy and its 3 subscores as floats between 0 and 1.
    """
    satb = Satb()
    if not satb.valid_chord(pred_next):
        return (0, 0, 0, 0)
    
    # get notes of chords, octave independent
    gt_set = set([note % 12 for note in gt_next])
    pred_set = set([note % 12 for note in pred_next])

    # score 
    notes_score = len(gt_set.intersection(pred_set)) / len(gt_set.union(pred_set))

    # score inversion (bass note)
    gt_bass = gt_next[3] % 12
    pred_bass = pred_next[3] % 12
    if pred_bass == gt_bass:
        inversion_score = 1
    elif pred_bass in gt_set:
        inversion_score = 0.25
    else:
        inversion_score = 0

    # score 3 upper voices
    voice_scores = np.zeros((3))
    for i in range(3):
        if pred_next[i] == gt_next[i]:
            voice_scores[i] = 1
        elif pred_next[i] in gt_set:
            voice_scores[i] = 0.75
    voicing_score = np.mean(voice_scores) 

    total_accuracy = (notes_score * NOTES_COEF) \
                    + (inversion_score * INVERSION_COEF) \
                    + (voicing_score * VOICING_COEF)                    

    return total_accuracy, notes_score, inversion_score, voicing_score

def accuracy_np(gt_next, pred_next):
    """
    Accuracy function for ndarrays of chords.

    Args:
        gt_next: ndarray of shape (X, 4), 
                where each row follows format described in `accuracy` function
        pred_next: same format as above

    Return:
        Total accuracy and its 3 subscores as ndarrays of shape (X, 1),
        where the ith row is the corresponding accuracy/ies for the ith
        chords in gt_next/pred_next
    """    
    if gt_next.shape[0] != pred_next.shape[0] or gt_next.shape[1] != 4 or pred_next.shape[1] != 4:
        raise ValueError("gt_next and pred_next must both have chord shape.", gt_next.shape, pred_next.shape)

    total_np = np.zeros((gt_next.shape[0]))
    notes_np = np.zeros_like(total_np)
    inversion_np = np.zeros_like(total_np)
    voicing_np = np.zeros_like(total_np)

    for i in range(gt_next.shape[0]):
        total, notes, inversion, voicing = accuracy(gt_next[i].tolist(), pred_next[i].tolist())
        total_np[i] = total
        notes_np[i] = notes
        inversion_np[i] = inversion
        voicing_np[i] = voicing

    return total_np, notes_np, inversion_np, voicing_np

def accuracy_score(gt_next, pred_next):
    """
    Accuracy function for ndarrays of chords that returns a single value (averaged total accuracy).
    Used to fit interface required by RandomizedSearchCV, which needs score as a single value.

    Args:
        gt_next: ndarray of shape (X, 4), 
                where each row follows format described in `accuracy` function
        pred_next: same format as above

    Return:
        Average total accuracy across all chords as a single float between 0 and 1.
    """    
    total_np, _, _, _ = accuracy_np(gt_next, pred_next)
    return np.mean(total_np)




    

