"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/30/2019
Filename: voice.py
Description: 
    A class to represent a generic voice.
"""

from utils.utils import note_to_num, num_to_note

class Voice():
    def __init__(self, low_note: str, high_note: str):
        self.low_note = low_note
        self.high_note = high_note

        self.low_num = note_to_num(low_note)
        self.high_num = note_to_num(high_note)
        
        self.range = self.high_num - self.low_num + 1  # number of notes in voice's range

    def in_range(self, note: int) -> bool:
        """Determine whether the given note is in the voice's range."""
        return (self.low_num <= note) and (note <= self.high_num)

    def transpose_range(self, note: int) -> tuple:
        """Determine the valid number of half-steps to tranpose the
        given note up and down while staying within the voice's range.

        Returns:
            Number of valid steps down (nonpositive)
            Number of valid steps up (nonnegative)
        """  
        if self.in_range(note):
            transpose_up = self.high_num - note
            transpose_down = self.low_num - note
            return transpose_down, transpose_up
        else:
            raise ValueError('Note is out-of-range', note, num_to_note(note))

    def normalize(self, note: int) -> float:
        """Convert the note to a float [0, 1] based on this voice's range"""
        if self.in_range(note):
            return (note - self.low_num) / self.range
        else:
            raise ValueError('Note is out-of-range', note, num_to_note(note)) 

    def scale(self, note: int) -> int:
        """Scale the note from [self.low_num, self.high_num] to [0, self.range)"""
        if self.in_range(note):
            return (note - self.low_num)
        else:
            raise ValueError('Note is out-of-range', note, num_to_note(note)) 

    def unscale(self, scaled_note: int) -> int:
        """Unscale the note from [0, self.range) to [self.low_num, self.high_num]"""
        if (scaled_note < self.range):
            return (scaled_note + self.low_num)
        else:
            raise ValueError('Note is not scaled', note, num_to_note(note)) 
