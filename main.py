from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)

# pickle dump for the sentiment_model.py to run
model = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))

# For images
imgFolder = os.path.join('static', 'images')

# Configuration to load the folder
app.config['UPLOAD_FOLDER'] = imgFolder

# Routes to different pages of the site
# Here two app routes will help in going to the same page with two links 
@app.route("/")
@app.route("/home")
def home():
  logo = os.path.join(app.config['UPLOAD_FOLDER'], 'Group 8.svg')
  popup = os.path.join(app.config['UPLOAD_FOLDER'], 'Group 3.svg')
  flasklogo = os.path.join(app.config['UPLOAD_FOLDER'], 'Group 9.svg')
  return render_template('home.html', logo_img = logo, pop_img = popup, flask_img = flasklogo)

# Route to the sentiment model 
@app.route('/predict', methods = ['POST'])
def predict():
  input_reviews = [x for x in request.form.values()]
  final = [np.arrays(input_reviews)]
  prediction = model.predict_proba(final)
  output = '{0:.{1}f}'.format(prediction[0][1], 1)

  if output < str(0.4):
    return render_template('home.html', pred = 'Ooops!! You have a negative review')
  elif str(0.4) < output < str(0.7):
    return render_template('home.html', pred = 'Umm-hmm!! You have an okayish review')
  elif str(0.7) < output < str(1.0):
    return render_template('home.html', pred = 'Woo-hoo!! You have an positive review')

# Used to run the app
app.run(host='0.0.0.0', port=81, debug=True)
