"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 8/1/2019
Filename: __init__.py
Description: 
    Initializes Flask app.
"""

import os

from flask import Flask

app = Flask(__name__)

from backend import routes