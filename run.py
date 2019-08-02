"""
Project: ML-Chord-Progressions
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 8/1/2019
Filename: run.py
Description: 
    [FOR DEVELOPMENT PURPOSES ONLY]
    Runs the Flask backend server.
"""
from backend import app
app.run(host='0.0.0.0', port=8080, debug=True)