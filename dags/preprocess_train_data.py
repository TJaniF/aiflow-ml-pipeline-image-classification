"""
### TITLE

DESCRIPTION
"""

from airflow import Dataset
from airflow.decorators import dag, task
from astro import sql as aql
from astro.sql import get_value_list
from astro.files import get_file_list
from astro.sql.table import Table
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator

import pandas as pd
from pendulum import datetime
import os
import logging
import requests
import numpy as np
from PIL import Image
import duckdb
import json
import pickle

task_logger = logging.getLogger("airflow.task")

TRAIN_FILEPATH = "include/train"
TEST_FILEPATH = "include/test"
FILESYSTEM_CONN_ID = "local_file_default"
DB_CONN_ID = "duckdb_default"
REPORT_TABLE_NAME = "reporting_table"
TEST_TABLE_NAME = "test_table"

S3_BUCKET_NAME = "myexamplebucketone"
S3_IN_FOLDER_NAME = "in_train_data"
S3_TRAIN_FOLDER_NAME = "train_data"
AWS_CONN_ID = "aws_default"
IMAGE_FORMAT = ".jpeg"
TRAIN_DATA_TABLE_NAME = "train_data"
DUCKDB_PATH = "include/duckdb_database"
DUCKDB_POOL_NAME = "duckdb_pool"


LABEL_TO_INT_MAP = {"glioma": 0, "meningioma": 1}


@dag(
    start_date=datetime(2023, 1, 1),
    schedule=[Dataset(f"s3://{S3_BUCKET_NAME}/{S3_TRAIN_FOLDER_NAME}/")],
    catchup=False,
)
def preprocess_train_data():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(
        task_id="end",
        outlets=[Dataset(f"duckdb://{DUCKDB_PATH}/{TRAIN_DATA_TABLE_NAME}")],
    )

    create_duckdb_pool = BashOperator(
        task_id="create_duckdb_pool",
        bash_command=f"airflow pools list | grep -q '{DUCKDB_POOL_NAME}' || airflow pools set {DUCKDB_POOL_NAME} 1 'Pool for duckdb'",
    )

    # ------------------------------- #
    # Load train file ref into DuckDB #
    # ------------------------------- #

    list_train_files = get_file_list(
        task_id="list_train_files",
        path=f"s3://{S3_BUCKET_NAME}/{S3_TRAIN_FOLDER_NAME}",
        conn_id=AWS_CONN_ID,
    )

    @task(pool=DUCKDB_POOL_NAME)
    def create_table(table_name, db_path):
        con = duckdb.connect(db_path)
        con.execute(
            f"""CREATE TABLE IF NOT EXISTS {table_name} 
            (image_s3_key TEXT PRIMARY KEY, timestamp DATETIME, label INT)"""
        )

    @task
    def get_train_labels(label_to_int_map, file_path):
        str_label = file_path.split("/")[-1].split(" ")[0]
        int_label = label_to_int_map[str_label]
        return int_label

    @task(pool=DUCKDB_POOL_NAME)
    def load_file_references_to_duckdb(db_path, table_name, data_to_insert, **context):
        image_s3_key = data_to_insert[0]
        label = data_to_insert[1]
        timestamp = context["ts"]
        con = duckdb.connect(db_path)
        con.execute(
            f"INSERT OR IGNORE INTO {table_name} (image_s3_key, timestamp, label) VALUES (?, ?, ?) ",
            (image_s3_key, timestamp, label),
        )

        con.close()

    train_labels = get_train_labels.partial(label_to_int_map=LABEL_TO_INT_MAP).expand(
        file_path=list_train_files.map(lambda x: x.path)
    )

    (
        start
        >> create_duckdb_pool
        >> list_train_files
        >> [
            train_labels,
            create_table(table_name=TRAIN_DATA_TABLE_NAME, db_path=DUCKDB_PATH),
        ]
        >> load_file_references_to_duckdb.partial(
            db_path=DUCKDB_PATH, table_name=TRAIN_DATA_TABLE_NAME
        ).expand(
            data_to_insert=list_train_files.map(lambda x: x.path).zip(train_labels)
        )
        >> end
    )


preprocess_train_data()
