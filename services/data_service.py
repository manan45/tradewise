from flask import Flask, request, jsonify
import threading
from app.core.interface_adapters.dhan import fetch_and_store_stock_data

app = Flask(__name__)


@app.route('/data', methods=['POST'])
def handle_data():
    data = request.json
    symbol = data.get('symbol')
    interval = data.get('interval')
    threading.Thread(target=fetch_and_store_stock_data, args=(symbol, interval)).start()
    return jsonify({"status": "success"}), 200
    data = request.json
    symbol = data.get('symbol')
    interval = data.get('interval')
    threading.Thread(target=fetch_and_store_stock_data, args=(symbol, interval)).start()
    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
