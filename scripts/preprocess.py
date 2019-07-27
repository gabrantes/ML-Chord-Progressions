"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/21/2019
Filename: preprocess.py
Description: 
    Preprocess dataset, converting notes into ints and creating augmentations to 
    increase overall size of dataset.
"""

import random
import math
import argparse
import csv
import numpy as np
from utils.utils import note_to_num, num_to_note
from utils.aug import augment
from utils.chorus.satb import Satb

VERBOSE = False

def preprocess(input_file: str, output_file: str):
    """
    Reformat and augment the dataset.
    Writes data to './data/chords.csv' by default.

    Args:   
        input_file: the filepath to the dataset
    """

    # getting chord progressions from input file
    progressions = []
    with open(input_file) as f:
        header = f.readline().strip()
        for line in f:
            if "/" not in line:  # ignore lines that are comments
                elements = line[:-1].split(',')  # remove newline, split elements
                progressions.append(elements)
    f.close()

    satb = Satb()

    for idx, prog in enumerate(progressions):
        # convert all notes to ints
        for i in range(len(prog)):
            if not prog[i].isdigit():  # if note
                prog[i] = note_to_num(prog[i])
            else:
                prog[i] = int(prog[i])
        # ensure all chords are valid
        if not satb.valid_chord(prog[5:9]):
            raise ValueError(
                "Current chord is invalid.",
                idx,
                prog[5:9],
                [num_to_note(el) for el in prog[5:9]]
                )
        if not satb.valid_chord(prog[12:16]):
            raise ValueError(
                "Next chord is invalid.",
                idx,
                prog[12:16],
                [num_to_note(el) for el in prog[12:16]]
                )
        
    # remove duplicates and keep track of unique progressions using a set
    progression_set = set([tuple(prog) for prog in progressions])
    progressions = [list(prog) for prog in progression_set]
    
    beg_num_chords = len(progressions)
    beg_inv_chords = 0
    beg_sev_chords = 0

    duplicate_chords = 0

    end_num_chords = beg_num_chords
    end_inv_chords = 0
    end_sev_chords = 0

    for i in range(len(progressions)):
        if VERBOSE:
            print(i)        

        aug_count = 10  # number of augmentations to create
        sev_chord = False
        inv_chord = False
        if progressions[i][3] == 1 or progressions[i][10] == 1:  # if seventh chord           
            aug_count = 40
            beg_sev_chords += 1
            end_sev_chords += 1
            sev_chord = True
        if progressions[i][4] != 0 or progressions[i][11] != 0:  # if chord not in root-position
            aug_count = 40
            beg_inv_chords += 1
            end_inv_chords += 1
            inv_chord = True

        # data augmentation
        for _ in range(aug_count):
            new_prog = augment(progressions[i])

            if tuple(new_prog) not in progression_set:
                progressions.append(new_prog)
                progression_set.add(tuple(new_prog))
                end_num_chords += 1
                if inv_chord:
                    end_inv_chords += 1
                if sev_chord:
                    end_sev_chords += 1
            else:
                duplicate_chords += 1
                continue

            if VERBOSE:
                print("Before:")
                print("\tKey: {}".format(num_to_note(progressions[i][0])))
                print("\t{}".format([num_to_note(el) for el in progressions[i][5:9]]))
                print("\t{}".format([num_to_note(el) for el in progressions[i][12:16]]))
                print("After:")
                print("\tKey: {}".format(num_to_note(new_prog[0])))
                print("\t{}".format([num_to_note(el) for el in new_prog[5:9]]))
                print("\t{}".format([num_to_note(el) for el in new_prog[12:16]]))
                print("\n")

    shuffle = True
    if shuffle:
        random.shuffle(progressions)

    with open(output_file, 'w') as output_csv:
        output_csv.write(header + '\n')
        writer = csv.writer(output_csv, delimiter=',', lineterminator='\n')
        writer.writerows(progressions)
    output_csv.close()
    
    print("\n************")

    print("\nSTART:")
    print("Total number of progressions:\t\t{}".format(beg_num_chords))
    print("Number of prog. with inverted chords:\t{}\t{}".format(beg_inv_chords, beg_inv_chords/beg_num_chords))
    print("Number of prog. with seventh chords:\t{}\t{}".format(beg_sev_chords, beg_sev_chords/beg_num_chords))

    print("\nDuplicate progressions:\t\t\t{}".format(duplicate_chords))
    print("(created during augmentation and removed)")
    
    print("\nEND:")
    print("Total number of progressions:\t\t{}".format(end_num_chords))
    print("Number of prog. with inverted chords:\t{}\t{}".format(end_inv_chords, end_inv_chords/end_num_chords))
    print("Number of prog. with seventh chords:\t{}\t{}".format(end_sev_chords, end_sev_chords/end_num_chords))

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(
        description='Transform notes into ints, and create augmentations \
            by transposing to random keys.'
        )
    parser.add_argument("--input", 
        help="Filepath to dataset. DEFAULT: ./data/seeds.csv",
        default="./data/seeds.csv")
    parser.add_argument("--output", 
        help="The output .csv. DEFAULT: ./data/chords.csv",
        default="./data/chords.csv")
    parser.add_argument("-v", "--verbose",
        help="Prints out chords for debugging. DEFAULT: False",
        action='store_true')

    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    preprocess(args.input, args.output)