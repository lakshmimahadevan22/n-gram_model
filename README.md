# Next Word Predictor using N-Gram 

This project implements a basic N-gram language model from scratch to predict the next word(s) in a sentence. It also includes a simple user interface built with Streamlit.


---

## Features

- Train an N-gram model (bigram, trigram, etc.) from Gutenberg corpus using NLTK.
- Predict the next word(s) given a prefix using n-gram probabilities.
- Implements Laplace smoothing to handle unseen sequences.
- Sentence generation from given prefix.
- Streamlit-based UI for easy interaction.

---

## Python Dependencies

pip install streamlit nltk

## Download the NLTK Corpora

import nltk

nltk.download('gutenberg')

nltk.download('punkt')


## Run the app

streamlit run app.py


## User Interface

![alt text](<Screenshot 2025-07-19 at 5.48.27 PM.png>)