"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 8/3/2019
Filename: predictor.py
Description: 
    Defines a Predictor class for custom prediction
    routines when deployed on Google Cloud AI Platform.
"""

import utils
import numpy as np
import pickle

class Predictor(object):
    """Interface for constructing custom predictors."""

    def __init__(self, model):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model

    def predict(self, instances, **kwargs):
        """Performs custom prediction.

        Instances are the decoded values from the request. They have already
        been deserialized from JSON.

        Args:
            instances: A list of prediction input instances.
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results. This list must
            be JSON serializable.
        """
        inputs = np.asarray(instances)

        # Input format
        # inputs[i] = [tonic, maj_min, 
        #               cur_s, cur_a, cur_t, cur_b, 
        #               next_degree, next_seventh, next_inversion]

        preprocessed_inputs = np.empty((inputs.shape[0], 18))
        
        preprocessed_inputs[:, 0:4] = inputs[:, 2:6]  # current chord voicings
        preprocessed_inputs[:, 4:6] = inputs[:, -2:]  # next chord seventh and inversion
        preprocessed_inputs[:, 6:] = utils.get_chord_notes_np(
            inputs[:, 0],   # tonic
            inputs[:, 1],   # maj_min
            inputs[:, -3],  # next chord degree
            inputs[:, -2]   # next chord seventh
        )

        outputs = np.asarray(self._model.predict(preprocessed_inputs), dtype=np.int8)

        processed_outputs = utils.num_to_note_np(
            inputs[:, 0],  # tonic
            inputs[:, 1],  # maj_min
            outputs        # model outputs (predicted chord voicings as ints)
            )

        return processed_outputs.tolist()

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of Predictor using the given path.

        Loading of the predictor should be done in this method.

        Args:
            model_dir: The local directory that contains the exported model
                file along with any additional files uploaded when creating the
                version resource.

        Returns:
            An instance implementing this Predictor class.
        """
        model_path = os.path.join(model_dir, 'model.pkl')
        model = pickle.load(model_path)

        return cls(model)