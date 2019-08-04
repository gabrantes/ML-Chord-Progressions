"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 8/3/2019
Filename: setup.py
Description: 
    Packages 'deploy/' as a module for deployment on
    Google Cloud Platform.
"""

from setuptools import setup

setup(
    name='chordnet_prediction_routine',
    version='0.6',
    packages=['deploy']
)

# Run python setup.py sdist --formats=gztar