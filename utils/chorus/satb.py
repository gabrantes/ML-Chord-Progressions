"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/30/2019
Filename: satb.py
Description: 
    A class to represent and encapsulate four (4) voices:
    soprano, alto, tenor, bass (SATB)
"""

from utils.chorus.voice import Voice
from utils.utils import num_to_note

class Satb():
    def __init__(self):
        self.voices = [
            Voice("C4", "G5"),  # soprano
            Voice("G3", "D5"),  # alto
            Voice("C3", "G4"),  # tenor
            Voice("E2", "C4")   # bass
            ]

    def valid_chord(self, chord: list) -> bool:
        """Determine whether the given chord is valid for the SATB choir.

        Args:
            chord: a chord of notes as ints: [soprano, alto, tenor, bass]

        Returns:
            True if valid, False otherwise.
        """
        if len(chord) != 4:
            return False

        # check range
        for voice, note in zip(self.voices, chord):
            if not voice.in_range(note):
                return False

        # check for voice crossing
        for i in range(3):
            if chord[i] < chord[i+1]:
                return False
        return True

    def transpose_range_chord(self, chord: list) -> tuple:
        """
        Determine the valid number of half-steps to tranpose the
        given chord up and down while staying within range of voices.

        Args:
            chord: a list of notes as ints: [soprano, alto, tenor, bass]

        Returns:
            Number of valid steps down (nonpositive)
            Number of valid steps up (nonnegative)
        """
        if self.valid_chord(chord):
            transpose_down = -self.voices[3].range - 1  # values outside the largest range
            transpose_up = self.voices[3].range + 1

            for voice, note in zip(self.voices, chord):                
                voice_down, voice_up = voice.transpose_range(note)
                transpose_up = min(transpose_up, voice_up)
                transpose_down = max(transpose_down, voice_down)
            return transpose_down, transpose_up
        else:
            raise ValueError("Chord is invalid", chord, [num_to_note(el) for el in chord])

    def transpose_range(self, *args) -> tuple:
        """
        Determine the valid number of half-steps to tranpose all
        given chords up and down while staying within range of voices.

        Args:
            *args: the chords, where each chord is a list of ints

        Returns:
            Number of valid steps down (nonpositive)
            Number of valid steps up (nonnegative)
        """
        transpose_down = -self.voices[3].range - 1
        transpose_up = self.voices[3].range + 1

        for chord in args:
            chord_down, chord_up = self.transpose_range_chord(chord)
            transpose_down = max(transpose_down, chord_down)
            transpose_up = min(transpose_up, chord_up)
        return transpose_down, transpose_up

    def normalize(self, chord: list) -> list:
        """Normalize every note in each chord to the corresponding voice's range"""
        if self.valid_chord(chord):
            norm_chord = [voice.normalize(note) for voice, note in zip(self.voices, chord)]
            return norm_chord
        else:
            raise ValueError("Chord is invalid", chord, [num_to_note(el) for el in chord])

    def scale(self, chord: list) -> list:
        """Scale every note in the chord to the corresponding voice's range"""        
        if self.valid_chord(chord):
            scaled_chord = [voice.scale(note) for voice, note in zip(self.voices, chord)]
            return scaled_chord
        else:
            raise ValueError("Chord is invalid", chord, [num_to_note(el) for el in chord])

    def unscale(self, chord: list) -> list:
        """Unscale every note in the chord to the corresponding voice's range"""
        unscaled_chord = [voice.unscale(note) for voice, note in zip(self.voices, chord)]
        if self.valid_chord(unscaled_chord):
            return unscaled_chord
        else:
            raise ValueError(
                    "Unscaled chord is invalid",
                    unscaled_chord,
                    [num_to_note(el) for el in unscaled_chord]
                )
