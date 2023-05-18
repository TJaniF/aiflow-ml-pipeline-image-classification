"""
### DAG that picks and deploys the best model according to a metric
"""

from airflow import Dataset as AirflowDataset
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from pendulum import datetime
import duckdb
from airflow.sensors.base import PokeReturnValue

from include.config_variables import (
    KEY_METRIC,
    KEY_METRIC_ASC_DESC,
    DUCKDB_POOL_NAME,
    DUCKDB_PATH
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
    @task(pool=DUCKDB_POOL_NAME)
    def pick_best_model_from_db(db_path):
        con = duckdb.connect(db_path)
        best_model_latest_test_set = con.execute(
            f"""SELECT model_name
                FROM model_results
                WHERE test_set_num = (SELECT MAX(test_set_num) FROM model_results)
                ORDER BY {KEY_METRIC} {KEY_METRIC_ASC_DESC}
                LIMIT 1;"""
        ).fetchall()[0][
            0
        ]
        con.close()

        return best_model_latest_test_set
    
    @task
    def deploy_model(model):
        print(model)

    (
        start
        >> ensure_baseline_ran()
        >> deploy_model(pick_best_model_from_db(db_path=DUCKDB_PATH))
        >> end
    )


deploy_best_model()
