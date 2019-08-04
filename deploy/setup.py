"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 8/3/2019
Filename: setup.py
Description: 
    Packages all necessary scripts/files for deployment on
    Google Cloud Platform.
"""

from setuptools import setup

setup(
    name='chordnet_prediction_routine',
    version='0.1',
    scripts=['predictor.py', 'utils.py']
)

# Run python setup.py sdist --formats=gztar