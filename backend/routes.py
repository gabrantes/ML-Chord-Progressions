"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 8/1/2019
Filename: routes.py
Description: 
    Declare REST endpoint(s).
"""

from flask import request, jsonify, abort
from backend import app

import numpy as np
import joblib
from utils.utils import get_chord_notes, num_to_note_key

MODEL_FILENAME = './model.joblib'

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
    try:
    # Calculating next chord notes
        next_chord_notes = get_chord_notes(
            data['tonic'], data['maj_min'], data['next_degree'], data['next_seventh']
            )
    except Exception as e:
        print(e)
        return None

    try:
        clf_input = np.empty((18))
        clf_input[0:4] = data['cur_chord']
        clf_input[5] = data['next_seventh']
        clf_input[6] = data['next_inversion']
        clf_input[6:] = next_chord_notes

        print("Loading model...")
        clf = joblib.load(MODEL_FILENAME)

        # Process predictions
        raw_pred = clf.predict(clf_input[np.newaxis, :])
        raw_pred = np.squeeze(raw_pred)
        pred = num_to_note_key(data['tonic'], data['maj_min'], raw_pred.tolist())
    except Exception as e:
        print(e)
        return None

    return {"pred_next": pred}