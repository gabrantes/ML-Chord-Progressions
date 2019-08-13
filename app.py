"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 8/12/2019
Filename: app.py
Description: 
    Defines the REST API backend using Flask.
"""

from flask import Flask, request, jsonify, abort

import numpy as np
import pickle
from utils.utils import get_chord_notes_np, num_to_note_np

MODEL_FILENAME = './model.pkl'

# Create Flask app
app = Flask(__name__)

@app.route('/api/wakeup')
def init():
    """ Jumpstart the server """
    return str(200)

@app.route('/api/predict', methods=['POST'])
def send_predictions():
    """ Handle the HTTP request """
    input_data = request.get_json()
    if input_data is None:
        abort(400, 'Received empty body or non-JSON body.')
    
    if not valid_input(input_data):    
        abort(400, 'Received JSON object of incorrect format.')

    output_data = get_predictions(input_data)

    return jsonify(output_data)

def valid_input(data: dict) -> bool:
    """ Validate the data """
    return ('instances' in data)

def get_predictions(data: dict) -> dict:
    """ Handles input/output to and from the random forest classifier """
    try:
        inputs = np.asarray(data['instances'], dtype=np.int8)

        # Input format
        # inputs[i] = [tonic, maj_min, 
        #               cur_s, cur_a, cur_t, cur_b, 
        #               next_degree, next_seventh, next_inversion]

        processed_inputs = np.empty((inputs.shape[0], 18))
        
        processed_inputs[:, 0:4] = inputs[:, 2:6]  # current chord voicings
        processed_inputs[:, 4:6] = inputs[:, -2:]  # next chord seventh and inversion
        processed_inputs[:, 6:] = get_chord_notes_np(
            inputs[:, 0],   # tonic
            inputs[:, 1],   # maj_min
            inputs[:, -3],  # next chord degree
            inputs[:, -2]   # next chord seventh
        )

        with open(MODEL_FILENAME, 'rb') as f:
            model = pickle.load(f)

        outputs = np.asarray(model.predict(processed_inputs), dtype=np.int8)

        processed_outputs = num_to_note_np(
            inputs[:, 0],  # tonic
            inputs[:, 1],  # maj_min
            outputs        # model outputs (predicted chord voicings as ints)
            )

        output_data = {"predictions": processed_outputs.tolist()}

    except Exception as e:
        output_data = {"error": str(e)}

    return output_data

if __name__ == "__main__":
    app.run()