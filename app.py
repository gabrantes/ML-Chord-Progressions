"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 8/1/2019
Filename: app.py
Description: 
    Defines the REST API backend using Flask.

    (Deprecated, abandoned in favor of deployment on
    Google Cloud Platform)
"""

from flask import Flask, request, jsonify, abort

import numpy as np
import pickle
from utils.utils import get_chord_notes, num_to_note_key

MODEL_FILENAME = 'model.pkl'

# Create Flask app
app = Flask(__name__)

@app.route('/api/')
def init():
    """ Jumpstart the server """
    return 200

@app.route('/api/predict', methods=['POST'])
def send_predictions():
    """ Handle the HTTP request """
    input_data = request.get_json()
    if input_data is None:
        abort(400, 'Received empty body or non-JSON body.')
    
    if not valid_input(input_data):    
        abort(400, 'Received JSON object with missing attributes.')

    output_data = get_predictions(input_data)

    if output_data is None:
        abort(400, 'Error thrown while getting prediction.')

    return jsonify(output_data)

def valid_input(data: dict) -> bool:
    """
    Validate the data by ensuring it contains all necessary
    attributes (keys).
    """
    attributes = [
        'tonic', 'maj_min', 'cur_chord',
        'next_degree', 'next_seventh', 'next_inversion'
    ]

    for attr in attributes:
        if attr not in data:
            return False
    return True

def get_predictions(data: dict) -> dict:
    """ Handles input/output to and from the random forest classifier """
    output_data = None

    try:
        # Calculating next chord notes
        next_chord_notes = get_chord_notes(
            data['tonic'], data['maj_min'], data['next_degree'], data['next_seventh']
            )

        clf_input = np.empty((18))
        clf_input[0:4] = data['cur_chord']
        clf_input[5] = data['next_seventh']
        clf_input[6] = data['next_inversion']
        clf_input[6:] = next_chord_notes

        print("Loading model...")
        clf = pickle.load(MODEL_FILENAME)

        # Process predictions
        print("Making predictions...")
        raw_pred = clf.predict(clf_input[np.newaxis, :])
        raw_pred = np.squeeze(raw_pred)
        print("Converting predictions...")
        pred = num_to_note_key(data['tonic'], data['maj_min'], raw_pred.tolist())

        output_data = {"pred_next": pred}
    except Exception as e:
        print(e)

    print("Returning predictions...")
    return output_data

if __name__ == "__main__":
    app.run()