"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/21/2019
Filename: utils.py
Description: 
    A set of various helper functions.
"""

import numpy as np

def note_to_num(note_str: str) -> int:
    """Convert a musical pitch from string representation to an integer.

    Args: 
        note_str: The string representation of a musical pitch, e.g. 'C4'.

    Returns: 
        The corresponding integer of the pitch.

    Raises:
        ValueError: if note is invalid.
    """
    note_map = {
        'Cbb': -2,    'Cb': -1,    'C': 0,     'C#': 1,     'C##': 2,
        'Dbb': 0,     'Db': 1,     'D': 2,     'D#': 3,     'D##': 4,
        'Ebb': 2,     'Eb': 3,     'E': 4,     'E#': 5,     'E##': 6,
        'Fbb': 3,     'Fb': 4,     'F': 5,     'F#': 6,     'F##': 7,
        'Gbb': 5,     'Gb': 6,     'G': 7,     'G#': 8,     'G##': 9,
        'Abb': 7,     'Ab': 8,     'A': 9,     'A#': 10,    'A##': 11,
        'Bbb': 9,     'Bb': 10,    'B': 11,    'B#': 12,    'B##': 13
    }
    if note_str[:-1] not in note_map:
        raise ValueError("Cannot convert invalid note to int.", note_str)
    note_int = note_map[note_str[:-1]]
    octave = int(note_str[-1])
    note_int = 12*octave + note_int
    return note_int

def num_to_note(note_int: int, custom_map=None) -> str:
    """
    Convert a musical pitch from integer representation to a string.
    "Merge" enharmonic equivalents, e.g. the integers for 'F#' and 'Gb'
    just become the string 'F#'.

    Args: 
        note_int: The integer representation of a musical pitch.

    Returns:
        The corresponding string for the pitch.
    """
    octave = str(note_int // 12)
    rev_note_map = {
        0: 'C',      1: 'C#',    2: 'D',
        3: 'D#',     4: 'E',     5: 'F',
        6: 'F#',     7: 'G',     8: 'G#',
        9: 'A',     10: 'A#',   11: 'B'
    }
    if custom_map is not None:
        rev_note_map.update(custom_map)

    note_str = rev_note_map[note_int % 12] + octave
    return note_str

def num_to_note_key(tonic: int, maj_min: int, chord) -> list:
    """
    Convert a chord from integer representation to a string.
    Determines enharmonic equivalent based on provided key (tonic and maj_min).

    Args: 
        tonic: (0 - 11), corresponding to (C - B)
        maj_min: 1 for major, 0 for minor
        chord: notes as ints

    Returns:
        [soprano, alto, tenor, bass] as strings
    """
    if tonic < 0 or tonic > 11:
        raise ValueError("Invalid key. Key must be in range (0, 11).", tonic)
    if maj_min < 0 or maj_min > 1:
        raise ValueError("Invalid quality. Must be 1 for major, 0 for minor.", maj_min)

    sig = (tonic, maj_min)
    custom_map = None
    
    if sig == (1, 1) or sig == (10, 0):  # DbM or Bbm
        custom_map = {
            1: 'Db',     3: 'Eb',    6: 'Gb',
            8: 'Ab',    10: 'Bb'
        }
    elif sig == (3, 1) or sig == (0, 0):  # if EbM or Cm
        custom_map = {
            3: 'Eb',     8: 'Ab',   10: 'Bb'
        }
    elif sig == (4, 1) or sig == (1, 0):  # if EM or C#m
        custom_map = {
            0: 'B#'
        }
    elif sig == (5, 1) or sig == (2, 0):  # if FM or Dm
        custom_map = {
            10: 'Bb'
        }
    elif sig == (6, 1) or sig == (3, 0):  # if F#M or Ebm (D#m)
        if sig == (6, 1):
            custom_map = {
                5: 'E#'
            }
        else:
            custom_map = {
                3: 'Eb',    6: 'Gb',
                8: 'Ab',    10: 'Bb'  
                # EDGE CASE: Cb
            }
    elif sig == (8, 1) or sig == (5, 0):  # if AbM or Fm
        custom_map = {
            1: 'Db',     3: 'Eb',    6: 'Gb',
            8: 'Ab',    10: 'Bb'
        }
    elif sig == (9, 1) or sig == (6, 0):  # if AM or F#m
        custom_map = {
            5: 'E#'
        }
    elif sig == (10, 1) or sig == (7, 0):  # if BbM or Gm
        custom_map = {
            3: 'Eb',    10: 'Bb'
        }
    elif sig == (11, 1) or sig == (8, 0):  # if BM or G#m
        custom_map = {
            7: 'F##'
        }

    return [num_to_note(note, custom_map=custom_map) for note in chord]

def num_to_note_np(tonic, maj_min, chords):
    """
    Convert chords from integer representation to a string.
    Determines enharmonic equivalents based on provided key (tonic and maj_min).

    Args:
        tonic: ndarray of tonics, shape = (X,)
        maj_min: ndarray of maj_mins, shape = (X,)
        chords: ndarray of chords, shape = (X, 4)
    """
    if not (tonic.shape == maj_min.shape and \
        tonic.shape[0] == chords.shape[0]):
        raise ValueError("All input ndarrays must have same size along axis 0.")
    
    converted_chords = np.empty_like(chords, dtype=np.dtype('U5'))

    for i in range(chords.shape[0]):
        converted_chords[i, :] = num_to_note_key(tonic[i], maj_min[i], chords[i, :])

    assert converted_chords.shape == chords.shape

    return converted_chords   

def convert_key(tonic, maj_min):
    """
    Convert tonic from integer representation to a string.

    Args:
        tonic: ndarray of tonics, shape = (X,)
        maj_min: ndarray of maj_mins, shape = (X,)
    """
    keys = num_to_note_np(tonic, maj_min, np.reshape(tonic, (tonic.shape[0], 1)))   
    keys = np.squeeze(keys)
    keys = np.where(
        maj_min,
        np.vectorize(lambda x: x[:-1] + "M")(keys),
        np.vectorize(lambda x: x[:-1] + "m")(keys)
    )
    return keys

def search_report(results, n_top=5):
    """ Utility function to report best scores from RandomizedSearchCV / GridSearchCV """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def get_chord_notes(tonic: int, maj_min: int, degree: int, seventh: int) -> list:
    """
    Given the key signature and degree/seventh of a chord, return the notes
    belonging to that chord (with notes as ints).
    """
    if maj_min == 1:
        # diatonic chords of major key
        note_dict = {
            1: [0, 4, 7, 11],
            2: [2, 5, 9, 0],
            3: [4, 7, 11, 2],
            4: [5, 9, 0, 4],
            5: [7, 11, 2, 5],
            6: [9, 0, 4, 7],
            7: [11, 2, 5, 9]
        }
    else:
        # diatonic chords of minor key
        note_dict = {
            1: [0, 3, 7, 10],
            2: [2, 5, 8, 0],
            3: [3, 7, 10, 2],
            4: [5, 8, 0, 3],
            5: [7, 11, 2, 5],
            6: [8, 0, 3, 7],
            7: [11, 2, 5, 8]
        }
    note_list = note_dict[degree]  # select chord degree
    note_list = [(note + tonic) for note in note_list]  # tranpose to specified tonic
    if seventh == 0:
        note_list = note_list[:-1]  # remove seventh note, if specified
    note_list = [(note % 12) for note in note_list]
    chord_notes = [0] * 12
    for note in note_list:
        chord_notes[note] = 1
    return chord_notes

def get_chord_notes_np(tonic, maj_min, degree, seventh):
    """
    Same as get_chord_notes except it accepts ndarrays as input and returns
    an ndarray as output.

    Args: each param is an ndarray of shape (X,)
    """
    if not (tonic.shape == maj_min.shape \
        and tonic.shape == degree.shape \
        and tonic.shape == seventh.shape):
        raise ValueError("All input ndarrays must have same size along axis 0.")

    chord_notes = np.zeros((tonic.shape[0], 12), dtype=np.int8)
    for i in range(chord_notes.shape[0]):
        chord_notes[i, :] = get_chord_notes(tonic[i], maj_min[i], degree[i], seventh[i])
    return chord_notes
