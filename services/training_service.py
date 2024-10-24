from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

def train_model_logic(data):
    # Implement model training logic here
    pass

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    threading.Thread(target=train_model_logic, args=(data,)).start()
    return jsonify({"status": "training started"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
