import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from app.core.ai.tradewise_ai import start_training
import asyncio

load_dotenv()

# app = Flask(__name__)

# @app.route('/train', methods=['POST'])
# def train_model():
#     start_training()
#     return jsonify({"status": "training started"}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.getenv('API_PORT')))


def train_model():
    asyncio.run(start_training("AAPL"))  # or any other stock symbol

if __name__ == '__main__':
    train_model()
