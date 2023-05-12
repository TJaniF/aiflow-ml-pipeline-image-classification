# S3 configurations
S3_BUCKET_NAME = "myexamplebucketone"

AWS_CONN_ID = "aws_default"
IMAGE_FORMAT = ".jp*g"

# normal vs glioma settings
S3_IN_TRAIN_FOLDER_NAME = "in_train_data_ng"
S3_TRAIN_FOLDER_NAME = "train_data_ng"
S3_IN_TEST_FOLDER_NAME = "in_test_data_ng"
S3_TEST_FOLDER_NAME = "test_data_ng"

# meningioma vs glioma settings
"""S3_IN_TRAIN_FOLDER_NAME = "in_train_data"
S3_TRAIN_FOLDER_NAME = "train_data"
S3_IN_TEST_FOLDER_NAME = "in_test_data"
S3_TEST_FOLDER_NAME = "test_data"""

# DuckDB configurations
DB_CONN_ID = "duckdb_default"
DUCKDB_PATH = "include/duckdb_database"
DUCKDB_POOL_NAME = "duckdb_pool"
TRAIN_DATA_TABLE_NAME = "train_data"
TEST_DATA_TABLE_NAME = "test_data"
RESULTS_TABLE_NAME = "model_results"

# sensor configurations
POKE_INTERVAL = 2

# ML related configurations
BASE_MODEL_NAME = "microsoft/resnet-50"
LOCAL_TEMP_TRAIN_FOLDER = "include/train"
LOCAL_TEMP_TEST_FOLDER = "include/test"
LABEL_TO_INT_MAP = {"normal": 0, "glioma": 1}
FINE_TUNED_MODEL_PATHS = "include/fine_tuned_models"

# Slack config
SLACK_CONNECTION_ID = "slack_default"
SLACK_CHANNEL = "alerts"
SLACK_MESSAGE = """
:tada: Model Test Successful :tada:

The {{ ti.task_id }} task finished testing the model: {{ ti.xcom_pull(task_ids='test_classifier')['model_name'] }}!

Fine-tuned model results:
Average test loss: {{ ti.xcom_pull(task_ids='test_classifier')['average_test_loss'] }}
Accuracy: {{ ti.xcom_pull(task_ids='test_classifier')['accuracy'] }}
Precision: {{ ti.xcom_pull(task_ids='test_classifier')['precision'] }}
Recall: {{ ti.xcom_pull(task_ids='test_classifier')['recall'] }}
F1-Score: {{ ti.xcom_pull(task_ids='test_classifier')['f1_score'] }}
AUC: {{ ti.xcom_pull(task_ids='test_classifier')['auc'] }}

Comparison:
Base Rate Accuracy of the test set: {{ var.value.get('baseline_accuracy', 'Not defined') }}
Pre-fine-tuning average test loss: {{ var.value.get('baseline_model_av_loss', 'Not defined') }}
Pre-fine-tuning test accuracy:  {{ var.value.get('baseline_model_accuracy', 'Not defined') }}
"""