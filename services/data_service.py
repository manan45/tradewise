from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

def process_data(data):
    # Implement data processing logic here
    pass

@app.route('/data', methods=['POST'])
def handle_data():
    data = request.json
    threading.Thread(target=process_data, args=(data,)).start()
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
