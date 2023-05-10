"""
### TITLE

DESCRIPTION
"""

from airflow import Dataset as AirflowDataset
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from pendulum import datetime
import logging
import duckdb
from airflow.sensors.base import PokeReturnValue


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
TEST_DATA_TABLE_NAME = "test_data"
DUCKDB_PATH = "include/duckdb_database"
DUCKDB_POOL_NAME = "duckdb_pool"

LABEL_TO_INT_MAP = {"glioma": 0, "meningioma": 1}
LOCAL_TEMP_TEST_FOLDER = "include/test"


@dag(
    start_date=datetime(2023, 1, 1),
    schedule=[AirflowDataset("new_model_tested")],
    catchup=False,
)
def deploy_best_model():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task.sensor(poke_interval=2, timeout=3600, mode="poke")
    def ensure_baseline_ran() -> PokeReturnValue:
        baseline_model_exists = bool(Variable.get("baseline_model_evaluated", False))
        return PokeReturnValue(is_done=baseline_model_exists, xcom_value="poke_return")

    @task(pool=DUCKDB_POOL_NAME)
    def pick_best_model_from_db(db_path):
        con = duckdb.connect(db_path)
        best_model_latest_test_set = con.execute(
            """SELECT model_name
                FROM model_results
                WHERE test_set_num = (SELECT MAX(test_set_num) FROM model_results)
                ORDER BY test_av_loss ASC
                LIMIT 1;""" # want higher false negative - be more sensitive recall!!. maybe optimized for f score (long discussion on medium), want to use precision or recall
        ).fetchall()[0][0] #ROC area
        con.close()

        return best_model_latest_test_set

    @task
    def deploy_model(model):
        pass

    (
        start
        >> ensure_baseline_ran()
        >> deploy_model(pick_best_model_from_db(db_path=DUCKDB_PATH))
        >> end
    )


deploy_best_model()
