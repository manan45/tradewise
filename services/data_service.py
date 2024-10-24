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

# Define an Airflow DAG for the data pipeline
dag = DAG('data_pipeline', description='Data Pipeline DAG',
          schedule_interval='@daily',
          start_date=datetime(2024, 10, 25), catchup=False)

data_pipeline_task = PythonOperator(task_id='data_pipeline_task', python_callable=data_pipeline, dag=dag)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
