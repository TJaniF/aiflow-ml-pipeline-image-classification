"""
### ML pipeline DAG that prepares a set of image paths and labels

This DAG fetches a list of file names from a specific S3 location and creates
a table in a local DuckDB to store these references. A separate task retrieves the
labels for the images and stores them in the same table.

Within the ML pipeline repository this DAG creates the table for the current train data.
"""

from airflow import Dataset
from airflow.decorators import dag, task
from astro.files import get_file_list
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator

from pendulum import datetime
import duckdb

from include.config_variables import (
    S3_BUCKET_NAME,
    S3_TRAIN_FOLDER_NAME,
    AWS_CONN_ID,
    DUCKDB_PATH,
    TRAIN_DATA_TABLE_NAME,
    DUCKDB_POOL_NAME,
    LABEL_TO_INT_MAP,
)


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

    # create an Airflow pool for all tasks writing to DuckDB (if it does not exist yet)
    # see also https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/pools.html
    create_duckdb_pool = BashOperator(
        task_id="create_duckdb_pool",
        bash_command=f"airflow pools list | grep -q '{DUCKDB_POOL_NAME}' || airflow pools set {DUCKDB_POOL_NAME} 1 'Pool for duckdb'",
    )

    # ------------------------------- #
    # Load train file ref into DuckDB #
    # ------------------------------- #

    # list the files in the S3 bucket
    list_train_files = get_file_list(
        task_id="list_train_files",
        path=f"s3://{S3_BUCKET_NAME}/{S3_TRAIN_FOLDER_NAME}",
        conn_id=AWS_CONN_ID,
    )

    # create a DuckDB table
    @task(pool=DUCKDB_POOL_NAME)
    def create_table(table_name, db_path):
        con = duckdb.connect(db_path)
        con.execute(
            f"""CREATE TABLE IF NOT EXISTS {table_name} 
            (image_s3_key TEXT PRIMARY KEY, timestamp DATETIME, label INT)"""
        )

    # retrieve labels, in this case from the file names
    @task
    def get_train_labels(label_to_int_map, file_path):
        str_label = file_path.split("/")[-1].split(" ")[0]
        int_label = label_to_int_map[str_label]
        return int_label

    # write the references to the training files to duckdb together with the correct label
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

    # call labelling function with mapped input (one task per file in S3)
    train_labels = get_train_labels.partial(label_to_int_map=LABEL_TO_INT_MAP).expand(
        file_path=list_train_files.map(lambda x: x.path)
    )

    # set Airflow dependencies
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
