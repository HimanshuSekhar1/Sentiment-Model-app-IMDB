from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)

imgFolder = os.path.join('static', 'images')

app.config['UPLOAD_FOLDER'] = imgFolder

# Routes to different pages of the site
# Here two app routes will help in going to the same page with two links 
@app.route("/")
@app.route("/home")
def home():
  logo = os.path.join(app.config['UPLOAD_FOLDER'], 'Group 8.png')
  return render_template('home.html', logo_img = logo)

# Used to run the app
app.run(host='0.0.0.0', port=81, debug=True)
