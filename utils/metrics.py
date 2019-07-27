"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/25/2019
Filename: metrics.py
Description: 
    Custom accuracy function for comparing predicted chord to ground truth chord.
"""

import numpy as np
import pandas as pd
from utils.chorus.satb import Satb

NOTES_COEF = 0.7
INVERSION_COEF = 0.2
VOICING_COEF = 0.1

def accuracy_np(gt_next, pred_next):    
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

def accuracy(gt_next, pred_next):
    """
    Custom accuracy function for comparing predicted chord to ground truth chord.

    Args:
        gt_next: [soprano, alto, tenor, bass] where each note is an int
        pred_next: same format as above

    Returns:
        Accuracy score as float between 0 and 1.
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




    

