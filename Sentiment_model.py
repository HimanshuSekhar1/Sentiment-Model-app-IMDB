# Import required modules

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import re

# Pickle package is for dumping all the codes into a single base. So when in the app new inputs will be done the app will not run the py code everytime for every inputs
import pickle

# To ignore any warnings that comes in the program
import warnings
warnings.filterwarnings("ignore")

# Tensorflow modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from google.colab import files
data = pd.read_csv('IMDB Dataset.csv')

# Remove punctuation
data['processed_review'] = data['review'].map(lambda x: re.sub('[,\.!?":-]', '', x))

# Remove  with space
data['processed_review'] = data['processed_review'].map(lambda x: re.sub('', ' ', x))

# Convert the titles to lowercase
data['processed_review'] = data['processed_review'].map(lambda x: x.lower())

# create new column for sentiments
data['num_sentiment'] = data['sentiment'].replace("positive", 1) # replaced positive with 1
data['num_sentiment'] = data['num_sentiment'].replace("negative", 0) # replaced negative with 0

# Check the data
# data.head()

# Creating the training subset 
training_size = 40000

training_reviews = data['processed_review'][0:training_size]
testing_reviews = data['processed_review'][training_size:]

training_labels = data['num_sentiment'][0:training_size]
testing_labels = data['num_sentiment'][training_size:]

# Various parameters for the process of tokenization
words = 1000 # vocab_size
oov_tok = ""
max_length = 1000
padding_type = 'post'
trunc_type = 'post'

# Creating tokenization
tokenizer = Tokenizer(num_words=words, oov_token=oov_tok)
tokenizer.fit_on_texts(training_reviews)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_reviews)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type) # This is for sequencing

testing_sequences = tokenizer.texts_to_sequences(testing_reviews)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Data training
embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(words, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics=['accuracy'])

history = model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels), verbose=1)

pickle.dump(model, open('sentiment_analysis_model.pkl', 'wb'))
model = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))
