from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from model import WasteClassifier
from utils import allowed_file, preprocess_image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classifier = WasteClassifier()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        processed = preprocess_image(path)
        prediction, confidence = classifier.predict(processed)

        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%"
        })

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    app.run(debug=True)
