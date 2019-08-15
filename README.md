# ML Chord Progressions

In traditional music theory, there are many rules, guidelines and exceptions when writing chord progressions for four voice parts. This is also referred to as four-part harmony. This project uses a random forest to predict chord voicings for four-part harmonies.

See the model in action here: [https://gabrantes.github.io/chordnet](https://gabrantes.github.io/chordnet)

&nbsp;

## Table of Contents

* [Getting Started](#getting-started)
   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
   * [Running](#running)
* [Dataset](#dataset)
* [Prediction](#prediction)
   * [Stage 1](#stage-1)
   * [Stage 2](#stage-2)
* [Training](#training)
* [Technologies](#technologies)
* [License](#license)

&nbsp;

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3
* pip OR pipenv

### Installation

1. Clone repository.

	 ```
   git clone https://github.com/gabrantes/ML-Chord-Progressions.git
   ``` 
  
2. Install requirements using pip OR pipenv.

   - Using pip:  
     ```
     pip install -r requirements.txt
     ```    
    
   - Using pipenv:  
     ```
     pipenv sync  
     ```
    
3. Add project to PYTHONPATH so modules can be found.

   ```
   set PYTHONPATH=path/to/ML-Chord-Progressions
   ```

### Running

1. Preprocess data and save result as `./data/chords.csv`.
   ```
   python scripts/preprocess.py
   ```
   
2. Train model and save predictions to `./output/output.csv`.
   ```
   python scripts/train.py
   ```

&nbsp;


## Dataset

Over 500 unique chord progressions were extracted from *Tonal Harmony (Kostka & Payne)*, a music theory textbook. To increase the amount of data, each chord progression was repeatedly transposed into different keys, generating a total of over 4000 unique chord progressions for training.

Categorical features were transformed using ordinal encoding. For example, chord degree was encoded as integers 1 through 7.

Musical pitches were also encoded as integers, given the fact that there are 12 half-steps in 1 octave. For example, C0 corresponds to 0, C#0 to 1, D0 to 2, ..., B0 to 11, C1 to 12, etc.

&nbsp;

## Prediction

To yield the highest accuracy, prediction was split into two stages.

### Stage 1

A function was created that takes as inputs:
* Key-signature
* Next chord degree
* Next chord seventh (a boolean specifiying whether it contained a seventh)

Given these inputs, it returned an array of 12 booleans, where the *i*th boolean represented the *i*th half-step/tone being present in that next chord.

For example, in C major, the notes in V7 are G, B, D and F. Converted to integers, these notes become G=7, B=11, D=2, and F=5. Then, the function would return:

[0 0 **1** 0 0 **1** 0 **1** 0 0 0 **1**]

which represents:

[C C# **D** D# E **F** F# **G** G# A A# **B**]

### Stage 2

Stage 2 makes use of the random forest classifier, which takes as inputs:
* Current chord voicings (soprano, alto, tenor, bass)
* Next chord seventh
* Next chord inversion
* The array of 12 booleans from stage 1

It then outputs an array of 4 integers (one for each voice), which represent the predicted notes of the next chord.

&nbsp;

## Training

Model hyperparameters were fine-tuned using repeated random search, followed by grid search.

&nbsp;

## Technologies

* Python (Sci-Kit Learn, Numpy, Pandas)

&nbsp;

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
