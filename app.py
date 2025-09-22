from flask import Flask, render_template, request, jsonify
import pickle, os
import numpy as np

BASE = os.path.dirname(__file__)
app = Flask(__name__)

with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

ACCURACY = 0.9807
MSE = 0.0193

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    if request.method == "POST":
        email_text = request.form["email"]
        if email_text.strip():
            X = vectorizer.transform([email_text])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0,1]
            prediction = int(pred)
            probability = float(proba)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        accuracy=ACCURACY,
        mse=MSE
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = data.get("text", "")
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0,1]
    return jsonify({"prediction": int(pred), "probability": float(proba)})

if __name__ == "__main__":
    app.run(debug=True)
