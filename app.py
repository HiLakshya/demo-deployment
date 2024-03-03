from flask import Flask, render_template, request, jsonify
from utils import model_predict
import pickle
import os


app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl",'rb'))
clf = pickle.load(open("models/clf.pkl",'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)  # Get data posted as a json
    email = data['content']
    prediction = model_predict(email)

    if prediction == 1:
        prediction = "Spam"
    else:
        prediction = "Not Spam"
        
    return jsonify({'prediction': prediction, 'email': email})  # Return prediction


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'ok'})

port = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)