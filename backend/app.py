import requests
from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
from transformers import pipeline
from dotenv import load_dotenv

# Load API keys
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Load AI model for fake news detection
fake_news_model = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-fake-news")

@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        news_text = data.get("news_text", "")

        if not news_text:
            return jsonify({"error": "No news text provided"}), 400

        result = fake_news_model(news_text)[0]
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
