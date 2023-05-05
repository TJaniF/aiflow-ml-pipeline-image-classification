"""
### TITLE

DESCRIPTION
"""

from airflow import Dataset
from airflow.decorators import dag
from astro.files import get_file_list
from astronomer.providers.amazon.aws.sensors.s3 import S3KeySensorAsync
from airflow.providers.amazon.aws.operators.s3 import (
    S3CopyObjectOperator,
    S3DeleteObjectsOperator,
)
from airflow.operators.empty import EmptyOperator
from pendulum import datetime
import logging


task_logger = logging.getLogger("airflow.task")

S3_BUCKET_NAME = "myexamplebucketone"
S3_IN_TEST_FOLDER_NAME = "in_test_data"
S3_TEST_FOLDER_NAME = "test_data"
AWS_CONN_ID = "aws_default"
IMAGE_FORMAT = ".jp*g"


@dag(
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
)
def in_new_test_data():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(
        task_id="end",
        outlets=[Dataset(f"s3://{S3_BUCKET_NAME}/{S3_TEST_FOLDER_NAME}/")],
    )

    wait_for_new_testing_data = S3KeySensorAsync(
        task_id="wait_for_new_testing_data",
        bucket_key=f"{S3_IN_TEST_FOLDER_NAME}/*{IMAGE_FORMAT}",
        bucket_name=S3_BUCKET_NAME,
        wildcard_match=True,
        aws_conn_id=AWS_CONN_ID,
        poke_interval=2,
    )

    in_file_list = get_file_list(
        path=f"s3://{S3_BUCKET_NAME}/{S3_IN_TEST_FOLDER_NAME}", conn_id=AWS_CONN_ID
    )

    def map_file_list_to_src_dest_key(astro_file_object):
        file_path = astro_file_object.path
        source_key = file_path
        dest_key = (
            f"s3://{S3_BUCKET_NAME}/{S3_TEST_FOLDER_NAME}/" + file_path.split("/")[-1]
        )
        return {"source_bucket_key": source_key, "dest_bucket_key": dest_key}

    copy_files = S3CopyObjectOperator.partial(
        task_id="copy_files",
        aws_conn_id=AWS_CONN_ID,
    ).expand_kwargs(in_file_list.map(map_file_list_to_src_dest_key))

    delete_files = S3DeleteObjectsOperator(
        task_id="delete_files",
        bucket=S3_BUCKET_NAME,
        prefix=f"{S3_IN_TEST_FOLDER_NAME}/",
        aws_conn_id=AWS_CONN_ID,
    )

    (
        start
        >> wait_for_new_testing_data
        >> in_file_list
        >> copy_files
        >> delete_files
        >> end
    )


in_new_test_data()
