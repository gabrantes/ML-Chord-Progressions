# ML Chord Progressions [WIP]

In traditional music theory, there are many guidelines and rules when writing chord progressions for four voice parts. This is also referred to as four-part harmony. This project develops a uses a multi-output Random Forest Classifier to take a current chord and return the correct voicings for the next chord.

"Tonal Harmony" (Kostka & Payne) is a popular music theory textbook that was used as the basis for this project. For example, the vocal ranges for each part were set as dictated in the book, and over 300 example chord progresions were used as 'seeds' for the dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3
* pip OR pipenv

### Installing

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
   
2. Train model and output predictions to `./output/combo_output.csv`.
   ```
   python scripts/test_combo.py
   ```

## Built With

* Python (Sci-Kit Learn, Numpy, Pandas)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
