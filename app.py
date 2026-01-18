from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from urllib.parse import urlparse

app = Flask(__name__)

# Load trained files
model = load_model("malicious_url_dnn_model.h5")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# Feature extraction
def extract_features(url):
    parsed = urlparse(url)

    base_features = {
        'url_length': len(url),
        'dots': url.count('.'),
        'hyphens': url.count('-'),
        'slashes': url.count('/'),
        'digits': sum(c.isdigit() for c in url),
        'at': url.count('@'),
        'question': url.count('?'),
        'equal': url.count('='),
        'https': 1 if parsed.scheme == 'https' else 0,
        'http': url.count('http')
    }

    row = []
    for col in feature_names:
        row.append(base_features.get(col, 0))  # missing features → 0

    return np.array(row).reshape(1, -1)

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        url = request.form['url']

        features = extract_features(url)
        features = scaler.transform(features)
        prediction = model.predict(features)[0][0]

        if prediction > 0.5:
            result = "⚠️ Malicious URL Detected"
        else:
            result = "✅ Safe URL"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
