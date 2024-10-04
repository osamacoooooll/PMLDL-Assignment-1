from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 3),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='ML Pipeline running every 5 minutes',
    schedule='*/5 * * * *',
)

def run_data_engineering():
    os.system('python3 /code/data/data.py')

def run_model_training():
    os.system('python3 /code/models/train_model.py') 

with dag:
    data_engineering_task = PythonOperator(
        task_id='data_engineering',
        python_callable=run_data_engineering,
    )

    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=run_model_training,
    )

    data_engineering_task >> model_training_task