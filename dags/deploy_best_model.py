"""
### DAG that picks and deploys the best model according to a metric
"""

from airflow import Dataset as AirflowDataset
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from astro import sql as aql
from airflow.models import Variable
from pendulum import datetime
from astro.sql.table import Table
from airflow.sensors.base import PokeReturnValue
import pandas as pd

from include.config_variables import (
    DUCKDB_POOL_NAME,
    DB_CONN_ID,
    RESULTS_TABLE_NAME,
)


@dag(
    start_date=datetime(2023, 1, 1),
    schedule=[AirflowDataset("new_model_tested")],
    catchup=False,
)
def deploy_best_model():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # make sure this DAG only runs if the baseline model has been evaluated
    @task.sensor(poke_interval=2, timeout=3600, mode="poke")
    def ensure_baseline_ran() -> PokeReturnValue:
        baseline_model_exists = bool(Variable.get("baseline_model_evaluated", False))
        return PokeReturnValue(is_done=baseline_model_exists, xcom_value="poke_return")

    # pick the best model from the duckdb records for the latest test set
    @aql.transform(pool=DUCKDB_POOL_NAME)
    def pick_best_model(in_table):
        return """SELECT model_name
                FROM {{ in_table }}
                WHERE test_set_num = (SELECT MAX(test_set_num) FROM {{ in_table }})
                ORDER BY auc DESC
                LIMIT 1;"""

    @aql.dataframe
    def deploy_model(df: pd.DataFrame):
        print(df["model_name"])

    model_deploy = deploy_model(
        df=pick_best_model(
            in_table=Table(conn_id=DB_CONN_ID, name=RESULTS_TABLE_NAME),
        )
    )

    aql.cleanup()

    (start >> ensure_baseline_ran() >> model_deploy >> end)


deploy_best_model()
