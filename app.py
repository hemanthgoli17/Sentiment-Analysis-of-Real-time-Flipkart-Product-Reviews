from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import joblib
import re

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return(render_template("home.html"))

@app.route("/prediction", methods=["POST"])
def prediction():
    review_text = request.form.get("review_text")
    data_point = np.array([review_text])
    model = joblib.load("best_models/naive_bayes.pkl") 
    prediction = model.predict(data_point)
    return(render_template("output.html", result = prediction))

if __name__ == '__main__':
    app.run(debug=True)