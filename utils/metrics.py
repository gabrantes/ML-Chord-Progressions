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

NOTE_COEF = 0.5
INVERSION_COEF = 0.3
VOICE_COEF = 0.2

def accuracy_df(gt_next, pred_next):
    all_metrics = accuracy_np(gt_next, pred_next)

    acc = pd.DataFrame(
        index=['total', 'note', 'inversion', 'voice'],
        columns=['mean', 'std', '25%', '50%', '75%']
        )
    acc['mean'] = [np.mean(x) for x in all_metrics]
    acc['std'] = [np.std(x) for x in all_metrics]
    acc['25%'] = [np.percentile(x, 25) for x in all_metrics]
    acc['50%'] = [np.percentile(x, 50) for x in all_metrics]
    acc['75%'] = [np.percentile(x, 75) for x in all_metrics]

    return acc

def accuracy_np(gt_next, pred_next):    
    if gt_next.shape[0] != pred_next.shape[0] or gt_next.shape[1] != 4 or pred_next.shape[1] != 4:
        raise ValueError("gt_next and pred_next must both have chord shape.", gt_next.shape, pred_next.shape)

    total_accuracy = np.zeros((gt_next.shape[0], 1))
    note_accuracy = np.zeros_like(total_accuracy)
    inversion_accuracy = np.zeros_like(total_accuracy)
    voice_accuracy = np.zeros_like(total_accuracy)

    for i in range(gt_next.shape[0]):
        total, note, inversion, voice = accuracy(gt_next[i].tolist(), pred_next[i].tolist())
        total_accuracy[i] = total
        note_accuracy[i] = note
        inversion_accuracy[i] = inversion
        voice_accuracy[i] = voice

    return total_accuracy, note_accuracy, inversion_accuracy, voice_accuracy    

def accuracy(gt_next, pred_next):
    """
    Custom accuracy function for comparing predicted chord to ground truth chord.

    Args:
        gt_next: [soprano, alto, tenor, bass] where each note is an int
        pred_next: same format as above

    Returns:
        Accuracy score as float between 0 and 1.
    """
    # get notes of chords, octave independent
    gt_set = set([note % 12 for note in gt_next])
    pred_set = set([note % 12 for note in pred_next])

    # score 
    note_score = len(gt_set.intersection(pred_set)) / len(gt_set.union(pred_set))

    # score inversion (bass note)
    gt_bass = gt_next[3] % 12
    pred_bass = gt_next[3] % 12
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
    voice_score = np.mean(voice_scores) 

    total_accuracy = (note_score * NOTE_COEF) \
                    + (inversion_score * INVERSION_COEF) \
                    + (voice_score * VOICE_COEF)                    

    return total_accuracy, note_score, inversion_score, voice_score




    

