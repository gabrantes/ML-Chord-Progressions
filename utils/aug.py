"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/21/2019
Filename: aug.py
Description: 
    Contains functions for augmenting the dataeset.
"""

from utils.chorus.satb import Satb
import numpy as np

def augment(progression: list) -> list:
    """
    `Augment` a progression by tranposing it to a new key.

    Args:
        progression: a list representing one line from the dataset
                    (with notes as ints)

    Returns:
        A list representing the new, tranposed progression
    """
    if len(progression) != 16:
        raise ValueError("Expected row from dataset to contain 16 elements.", len(progression))

    satb = Satb()
    lo, hi = satb.transpose_range(progression[5:9], progression[12:16])
    shift = np.random.randint(lo, hi+1)

    new_progression = progression[:]            

    # tonic (key signature)
    cur_key = new_progression[0]
    new_progression[0] = (cur_key + shift) % 12

    # cur chord
    for j in range(5, 9):
        new_progression[j] += shift

    # next chord
    for j in range(12, 16):
        new_progression[j] += shift
    
    return new_progression