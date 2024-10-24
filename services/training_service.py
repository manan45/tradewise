from flask import Flask, request, jsonify
import threading
from app.mongodb_client import MongoDBClient

app = Flask(__name__)

def train_model_logic():
    mongo_client = MongoDBClient(uri="mongodb://mongo:27017")
    stock_data = mongo_client.get_all_stocks()
    # Implement model training logic here using stock_data
    # Store results back in MongoDB if needed

@app.route('/train', methods=['POST'])
def train_model():
    threading.Thread(target=train_model_logic).start()
    return jsonify({"status": "training started"}), 200

# Define an Airflow DAG for the training pipeline
dag = DAG('training_pipeline', description='Training Pipeline DAG',
          schedule_interval='@daily',
          start_date=datetime(2024, 10, 25), catchup=False)

training_pipeline_task = PythonOperator(task_id='training_pipeline_task', python_callable=training_pipeline, dag=dag)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
