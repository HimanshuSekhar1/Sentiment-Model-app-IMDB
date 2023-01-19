from flask import Flask, render_template, request, url_for, redirect, session, flash
import os
import pickle
import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)

# pickle dump for the sentiment_model.py to run
# model = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))

# For images
imgFolder = os.path.join('static', 'images')

# Configuration to load the folder
app.config['UPLOAD_FOLDER'] = imgFolder

def init():
  global model, graph
  model = load_model('Sentiment_model.h5')
  graph = tf.compat.v1.get_default_graph()

# Routes to different pages of the site
# Here two app routes will help in going to the same page with two links 
@app.route("/", methods = ['GET', 'POST'])
@app.route("/home", methods = ['GET', 'POST'])
def home():
  logo = os.path.join(app.config['UPLOAD_FOLDER'], 'Group 8.svg')
  popup = os.path.join(app.config['UPLOAD_FOLDER'], 'Group 3.svg')
  flasklogo = os.path.join(app.config['UPLOAD_FOLDER'], 'Group 9.svg')
  return render_template('home.html', logo_img = logo, pop_img = popup, flask_img = flasklogo)

# Route to the sentiment model 
@app.route('/predict', methods = ['POST', 'GET'])
def predict_reviews():
  if request.method == 'POST':
    sentence = request.form['text']
    words = 1000
    oov_tok = "<OOV>"
    max_length = 1000

    tokenizer = Tokenizer(num_words = 1000, oov_token=oov_tok)
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen = 1000, padding='post', truncating='post')
    
    with graph.as_default():
      probability = model.predict(padded)
      output = '{0:.{1}f}'.format(probability[0][1], 1)
    if output < str(0.4):
      return render_template('home.html', pred = 'Ooops!! You have a negative review')
    elif str(0.4) < output < str(0.7):
      return render_template('home.html', pred = 'Umm-hmm!! You have an okayish review')
    elif str(0.7) < output < str(1.0):
      return render_template('home.html', pred = 'Woo-hoo!! You have a positive review')

# Used to run the app
if __name__ == "__main__":
  init()
  app.run(host='0.0.0.0', port=81, debug=True)
