from airflow.models.baseoperator import BaseOperator
from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification
import os
import numpy as np
from torch import tensor, device as tdevice, cuda, round as tround, sigmoid, no_grad
from torch.nn import BCELoss, Linear, BCEWithLogitsLoss
from torch.optim import Adam 
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_classification_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate evaluation metrics for binary classification.

    :param y_true: List or array of true binary labels.
    :param y_pred: List or array of predicted binary labels.
    :param threshold: Threshold value to use for binarizing predictions (default=0.5).

    Returns: Dictionary containing calculated metrics (precision, recall, accuracy, F1-score, false positives,
    false negatives, true positives, true negatives, and AUC).
    """
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)

    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    auc = roc_auc_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "auc": auc,
    }


class CustomImageDataset(Dataset):
    """
    Creates a pytorch image dataset from input images and labels.

    :param images_paths: list of paths to images as list of strings.
    :param labels: list of labels as list of integers or floats.
    :param transform_function: callable to be applied to images.
    """

    def __init__(
        self,
        images_paths: list,
        labels: list,
        transform_function: callable = None,
    ):
        self.images_paths = images_paths
        self.labels = labels
        self.transform_function = transform_function

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        label = self.labels[idx]
        label_tensor = tensor(label)

        image = Image.open(image_path)

        if self.transform_function:
            image = self.transform_function(image)

        return image, label_tensor


class FineTuneHuggingFaceBinaryImageClassifierOperator(BaseOperator):
    """
    Trains a binary HuggingFace image classification model on a list of locally saved images.

    :param model_name: name of the model to use as a string. Can reference a public HuggingFace model
    or be the path to a locally saved model.
    :param criterion: loss function. Default: BCEWithLogitsLoss().
    :param optimizer: model optimizer. Default: Adam.
    :param local_images_filepaths: list of paths to the training images (list of str).
    :param labels: list of labels for the training set (list of floats or ints).
    :param learning_rate: learning rate as a float.
    :param model_save_dir: directory to save the trained model to.
    :param train_transform_function: transform function for training images.
    :param batch_size: size of each training batch.
    :param num_epochs: number of epochs.
    :param shuffle: whether or not batches should be shuffled.
    :param num_workers_data_loader: number of workers to be used by the data loader.
    :param ignore_mismatched_sizes_resnet: whether mismatches with the size of the resnet should be ignored.
    """

    ui_color = "#91ed9d"

    template_fields = (
        "model_name",
        "criterion",
        "optimizer",
        "local_images_filepaths",
        "labels",
        "learning_rate",
        "model_save_dir",
        "train_transform_function",
        "batch_size",
        "num_epochs",
        "shuffle",
        "num_workers_data_loader",
        "ignore_mismatched_sizes_resnet",
    )

    def __init__(
        self,
        model_name: str,
        local_images_filepaths: list,
        labels: list,
        optimizer=Adam,
        criterion = BCEWithLogitsLoss(),
        learning_rate: float = 0.001,
        model_save_dir: str = "/",
        train_transform_function: callable = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        shuffle: bool = True,
        num_workers_data_loader: int = 0,
        ignore_mismatched_sizes_resnet: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.criterion = criterion
        self.optimizer = optimizer
        self.local_images_filepaths = local_images_filepaths
        self.labels = labels
        self.learning_rate = learning_rate
        self.model_save_dir = model_save_dir
        self.train_transform_function = train_transform_function
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.num_workers_data_loader = num_workers_data_loader
        self.ignore_mismatched_sizes_resnet = ignore_mismatched_sizes_resnet
        self.num_classes = 1

    def execute(self, context):
        # loading the train set from list of image paths and list of labels
        train_dataset = CustomImageDataset(
            images_paths=self.local_images_filepaths,
            labels=self.labels,
            transform_function=self.train_transform_function,
        )

        print(f"Successfully created the Train Dataset! Length: {len(train_dataset)}")

        # create the train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers_data_loader,
        )

        print(f"Successfully created the Train DataLoader!")

        # fetch model
        model = ResNetForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=self.ignore_mismatched_sizes_resnet,
        )

        print(f"Fetched model: {self.model_name}")

        model.classifier[-1] = Linear(
            model.classifier[-1].in_features, self.num_classes
        )

        print(f"Model target set to {self.num_classes} classes.")

        # freeze all layers except for the final ones
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        # initialize optimizer to only train final layers
        optimizer = self.optimizer(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate
        )

        print("All layers except final ones frozen!")

        # set device
        device = tdevice("cuda" if cuda.is_available() else "cpu")
        model = model.to(device)

        for epoch in range(self.num_epochs):
            for num_step, batch in enumerate(train_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                logits = outputs.logits.squeeze(-1)
                loss = self.criterion(logits, labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                predictions = tround(sigmoid(logits))

                print(f"Epoch: {epoch} / Step: {num_step}  loss: {loss.item()}")
                print(
                    f"    Predictions: {[int(x) for x in predictions.cpu().detach().numpy()]}"
                )
                print(
                    f"    True labels: {[int(x) for x in labels.cpu().detach().numpy()]}"
                )

        # trainer.save_model("test_trainer/model1")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        model.save_pretrained(self.model_save_dir)

        print(f"Model saved to {self.model_save_dir}")


class TestHuggingFaceBinaryImageClassifierOperator(BaseOperator):
    """
    Tests a binary HuggingFace image classification model on a list of locally saved images.

    :param model_name: name of the model to use as a string. Can reference a public HuggingFace model
    or be the path to a locally saved model.
    :param criterion: loss function. Default BCELoss().
    :param local_images_filepaths: list of paths to the testing images (list of str).
    :param labels: list of labels for the testing set (list of floats or ints).
    :param train_transform_function: transform function for training images.
    :param batch_size: size of each training batch.
    :param shuffle: whether or not batches should be shuffled.
    :param num_workers_data_loader: number of workers to be used by the data loader.
    :param ignore_mismatched_sizes_resnet: whether mismatches with the size of the resnet should be ignored.
    """

    ui_color = "#ebab34"

    template_fields = (
        "model_name",
        "criterion",
        "local_images_filepaths",
        "labels",
        "test_transform_function",
        "batch_size",
        "shuffle",
        "num_workers_data_loader",
        "ignore_mismatched_sizes_resnet",
    )

    def __init__(
        self,
        model_name: str,
        local_images_filepaths: list,
        labels: list,
        criterion = BCELoss(),
        test_transform_function: callable = None,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers_data_loader: int = 0,
        ignore_mismatched_sizes_resnet: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.criterion = criterion
        self.local_images_filepaths = local_images_filepaths
        self.labels = labels
        self.test_transform_function = test_transform_function
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers_data_loader = num_workers_data_loader
        self.ignore_mismatched_sizes_resnet = ignore_mismatched_sizes_resnet
        self.num_classes = 1

    def execute(self, context=None):
        # loading the test set from list of image paths and list of labels
        test_dataset = CustomImageDataset(
            images_paths=self.local_images_filepaths,
            labels=self.labels,
            transform_function=self.test_transform_function,
        )

        print(f"Successfully created the Test Dataset! Length: {len(test_dataset)}")

        # create the train loader

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers_data_loader,
        )

        print(f"Successfully created the Test DataLoader!")

        # fetch model
        model = ResNetForImageClassification.from_pretrained(
            self.model_name, ignore_mismatched_sizes=self.ignore_mismatched_sizes_resnet
        )

        print(f"Fetch model: {self.model_name}")

        model.classifier[-1] = Linear(
            model.classifier[-1].in_features, self.num_classes
        )

        print(f"Model target set to {self.num_classes} classes.")

        # set device
        device = tdevice("cuda" if cuda.is_available() else "cpu")
        model = model.to(device)

        # start model evaluation
        model.eval()

        test_loss = 0
        predictions = []
        true_labels = []

        print("Starting model evaluation...")

        with no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()

                outputs = model(images)
                probabilities = sigmoid(outputs.logits)

                predictions += (probabilities > 0.5).float().cpu().numpy().tolist()
                true_labels += labels.cpu().numpy().tolist()
                
        average_test_loss = test_loss / len(test_loader)
        labels = np.array(self.labels)
        metrics = calculate_binary_classification_metrics(true_labels, predictions)
        print(predictions)
        print(true_labels)
        print(f"Test Loss: {average_test_loss:.4f}, Metrics: {metrics}")

        context["ti"].xcom_push(key="predictions", value=predictions)

        return {
            "model_name": self.model_name,
            "timestamp": context["ts"],
            "average_test_loss": average_test_loss,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "true_positives": metrics["true_positives"],
            "true_negatives": metrics["true_negatives"],
            "auc": metrics["auc"],
        }
