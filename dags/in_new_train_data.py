"""
### EL DAG continuously moving files in S3

This DAG runs continuously and uses an async sensor to wait for a file of a specific
image to drop into an S3 folder. If a new file is detected it will be moved into a 
different S3 folder. 
Within the ML pipeline repository this DAG waits for new train data.
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

# import variables from local config file
from include.config_variables import (
    S3_BUCKET_NAME,
    S3_IN_TRAIN_FOLDER_NAME,
    S3_TRAIN_FOLDER_NAME,
    AWS_CONN_ID,
    IMAGE_FORMAT,
    POKE_INTERVAL,
)


@dag(
    start_date=datetime(2023, 1, 1),
    schedule="@continuous",
    max_active_runs=1,
    catchup=False,
)
def in_new_train_data():
    start = EmptyOperator(task_id="start")
    # the last task in the DAG will update the relevant Dataset to kick off
    # downstream DAGs in a data-driven way
    end = EmptyOperator(
        task_id="end",
        outlets=[Dataset(f"s3://{S3_BUCKET_NAME}/{S3_TRAIN_FOLDER_NAME}/")],
    )

    # asynchronously wait for a file to drop in S3
    wait_for_new_training_data = S3KeySensorAsync(
        task_id="wait_for_training_data",
        bucket_key=f"{S3_IN_TRAIN_FOLDER_NAME}/*{IMAGE_FORMAT}",
        bucket_name=S3_BUCKET_NAME,
        wildcard_match=True,
        aws_conn_id=AWS_CONN_ID,
        poke_interval=POKE_INTERVAL,
    )

    # get a list of all files at a specific S3 key using the Astro Python SDK
    in_file_list = get_file_list(
        path=f"s3://{S3_BUCKET_NAME}/{S3_IN_TRAIN_FOLDER_NAME}", conn_id=AWS_CONN_ID
    )

    # create dynamic arguments from the list of files
    def map_file_list_to_src_dest_key(astro_file_object):
        file_path = astro_file_object.path
        source_key = file_path
        dest_key = (
            f"s3://{S3_BUCKET_NAME}/{S3_TRAIN_FOLDER_NAME}/" + file_path.split("/")[-1]
        )
        return {"source_bucket_key": source_key, "dest_bucket_key": dest_key}

    # copy files from one location in S3 to another, dynamically mapped for one task per file
    copy_files = S3CopyObjectOperator.partial(
        task_id="copy_files",
        aws_conn_id=AWS_CONN_ID,
    ).expand_kwargs(in_file_list.map(map_file_list_to_src_dest_key))

    # delete files from ingestion folder
    delete_files = S3DeleteObjectsOperator(
        task_id="delete_files",
        bucket=S3_BUCKET_NAME,
        prefix=f"{S3_IN_TRAIN_FOLDER_NAME}/",
        aws_conn_id=AWS_CONN_ID,
    )

    # setting Airflow dependencies
    (
        start
        >> wait_for_new_training_data
        >> in_file_list
        >> copy_files
        >> delete_files
        >> end
    )


in_new_train_data()
