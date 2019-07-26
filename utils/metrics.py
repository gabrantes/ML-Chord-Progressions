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

SET_COEF = 0.4
VOICE_COEF = 0.6

def accuracy_df(gt_next, pred_next):
    all_metrics = accuracy_np(gt_next, pred_next)

    acc = pd.DataFrame(
        index=['total', 'voice', 'set'],
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
    voice_accuracy = np.zeros_like(total_accuracy)
    set_accuracy = np.zeros_like(total_accuracy)
    for i in range(gt_next.shape[0]):
        total, voice, set_ = accuracy(gt_next[i].tolist(), pred_next[i].tolist())
        total_accuracy[i] = total
        voice_accuracy[i] = voice
        set_accuracy[i] = set_
    return total_accuracy, voice_accuracy, set_accuracy    

def accuracy(gt_next, pred_next):
    """
    Custom accuracy function for comparing predicted chord to ground truth chord.

    Args:
        gt_next: [soprano, alto, tenor, bass] where each note is an int
        pred_next: same format as above

    Returns:
        Accuracy score as float between 0 and 1.
    """
    gt_set = set([note % 12 for note in gt_next])
    pred_set = set([note % 12 for note in pred_next])

    # intersection over union
    set_score = len(gt_set.intersection(pred_set)) / len(gt_set.union(pred_set))

    voice_scores = [0] * 4
    voice_coefs = [0.1, 0.1, 0.1, 0.7]  # weight bass more because bass note also determines inversion

    for i in range(3):
        if pred_next[i] == gt_next[i]:
            voice_scores[i] = 1
        elif pred_next[i] in gt_set:
            voice_scores[i] = 0.5

    # score bass note differently
    if pred_next[3] == gt_next[3]:
        voice_scores[3] = 1
        
    voice_scores = [w*x for (w, x) in zip(voice_coefs, voice_scores)]
    voice_total = sum(voice_scores)


    total_accuracy = (voice_total * VOICE_COEF) + (set_score * SET_COEF)
    return total_accuracy, voice_total, set_score




    

