from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    # Train model
    return jsonify({"status": "training started"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
