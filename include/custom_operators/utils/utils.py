from torchvision import transforms
import duckdb 
from airflow.models import Variable


def standard_transform_function(image):
    """A function normalizing images to the same size,
    handling RGB and Grey-scale inputs."""

    if image.mode == "RGB":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif image.mode == "L":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        raise ValueError("Unsupported image mode: %s" % image.mode)

    return transform(image)


def write_all_model_metrics_to_duckdb(db_path, table_name, **context):
    model_name = context["ti"].xcom_pull(task_ids="test_classifier")["model_name"]
    timestamp = context["ti"].xcom_pull(task_ids="test_classifier")["timestamp"]
    average_test_loss = context["ti"].xcom_pull(task_ids="test_classifier")[
        "average_test_loss"
    ]
    precision = context["ti"].xcom_pull(task_ids="test_classifier")["precision"]
    recall = context["ti"].xcom_pull(task_ids="test_classifier")["recall"]
    accuracy = context["ti"].xcom_pull(task_ids="test_classifier")["accuracy"]
    f1_score = context["ti"].xcom_pull(task_ids="test_classifier")["f1_score"]
    false_positives = context["ti"].xcom_pull(task_ids="test_classifier")[
        "false_positives"
    ]
    false_negatives = context["ti"].xcom_pull(task_ids="test_classifier")[
        "false_negatives"
    ]
    true_positives = context["ti"].xcom_pull(task_ids="test_classifier")[
        "true_positives"
    ]
    true_negatives = context["ti"].xcom_pull(task_ids="test_classifier")[
        "true_negatives"
    ]
    auc = context["ti"].xcom_pull(task_ids="test_classifier")["auc"]

    con = duckdb.connect(db_path)
    test_set_num = Variable.get("test_set_num")

    con.execute(
        f"""CREATE TABLE IF NOT EXISTS {table_name} (
                model_name TEXT PRIMARY KEY, 
                timestamp DATETIME, 
                average_test_loss FLOAT, 
                precision FLOAT,
                recall FLOAT, 
                accuracy FLOAT,
                f1_score FLOAT,
                fp INT,
                fn INT, 
                tp INT,
                tn INT,
                auc FLOAT,
                test_set_num INT
            )"""
    )

    con.execute(
        f"""INSERT OR REPLACE INTO {table_name} (
                model_name, timestamp, average_test_loss, precision, recall, 
                accuracy, f1_score, fp, fn, tp, tn, auc, test_set_num
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model_name,
            timestamp,
            float(average_test_loss),
            float(precision),
            float(recall),
            float(accuracy),
            float(f1_score),
            int(false_positives),
            int(false_negatives),
            int(true_positives),
            int(true_negatives),
            float(auc),
            test_set_num,
        ),
    )

    con.close()
