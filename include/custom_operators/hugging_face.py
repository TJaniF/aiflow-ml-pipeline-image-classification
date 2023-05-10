from airflow.models.baseoperator import BaseOperator
import torch
from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def transform_function(image):
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


def test_transform_function(image):
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


class CustomImageDataset(Dataset):
    def __init__(
        self,
        images_paths,
        labels,
        transform_function,
    ):
        self.images_paths = images_paths
        self.labels = labels
        self.transform_function = transform_function

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)

        if self.transform_function:
            image = self.transform_function(image)
        return image, torch.tensor(label, dtype=torch.long)


class TrainHuggingFaceImageClassifierOperator(BaseOperator):
    """
    Trains a HuggingFace image classification model on a list of locally saved images.

    :param model_name: name of the model to use
    :param local_images_filepaths: list of paths to the training images
    :param labels: list of labels for the training set
    :param batch_size: size of each training batch

    """

    ui_color = "#91ed9d"

    template_fields = (
        "model_name",
        "criterion",
        "local_images_filepaths",
        "labels",
        "num_classes",
        "model_save_dir",
        "train_transform_function",
        "batch_size",
        "shuffle",
    )

    def __init__(
        self,
        model_name: str,
        criterion,
        optimizer,
        local_images_filepaths: list,
        labels: list,
        num_classes: int,
        model_save_dir: str = "/",
        train_transform_function: callable = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        shuffle: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.criterion = criterion
        self.optimizer = optimizer #change optimizer
        self.local_images_filepaths = local_images_filepaths
        self.labels = labels
        self.num_classes = num_classes
        self.model_save_dir = model_save_dir
        self.train_transform_function = train_transform_function
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle = shuffle

    def execute(self, context):
        train_dataset = CustomImageDataset(
            images_paths=self.local_images_filepaths,
            labels=self.labels,
            transform_function=self.train_transform_function,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
        )

        # figure out how fine tuning happens inside hugging face
        model = ResNetForImageClassification.from_pretrained(self.model_name) # does this do something optimized for fine tune or not?
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, self.num_classes
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = self.optimizer(model.parameters(), lr=1e-4) ### lr is a hyperparameter learning rate to be adjusted. add as a parameter

        for epoch in range(self.num_epochs): #num of epochs
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    logits = outputs.logits
                    _, preds = torch.max(logits, 1)
                    loss = self.criterion(logits, labels)

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)

            print(
                f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
            )

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        model.save_pretrained(self.model_save_dir)


class TestHuggingFaceImageClassifierOperator(BaseOperator):
    """
    Tests a HuggingFace image classification model on a list of locally saved images.


    """

    ui_color = "#ebab34"

    template_fields = (
        "model_name",
        "criterion",
        "local_images_filepaths",
        "labels",
        "num_classes",
        "model_save_dir",
        "test_transform_function",
        "batch_size",
        "shuffle",
    )

    def __init__(
        self,
        model_name: str,
        criterion,
        local_images_filepaths: list,
        labels: list,
        num_classes: int,
        model_save_dir: str = "/",
        test_transform_function: callable = None,
        batch_size: int = 32,
        shuffle: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.criterion = criterion
        self.local_images_filepaths = local_images_filepaths
        self.labels = labels
        self.num_classes = num_classes
        self.model_save_dir = model_save_dir
        self.test_transform_function = test_transform_function
        self.batch_size = batch_size
        self.shuffle = shuffle

    def execute(self, context):
        test_dataset = CustomImageDataset(
            images_paths=self.local_images_filepaths,
            labels=self.labels,
            transform_function=self.test_transform_function,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
        )

        model = ResNetForImageClassification.from_pretrained(
            self.model_name, ignore_mismatched_sizes=True
        )
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, self.num_classes
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        model.eval()

        test_loss = 0
        correct = 0
        total = 0
        predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = self.criterion(outputs.logits, labels)

                test_loss += loss.item()

                _, predicted = torch.max(outputs.logits, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions += predicted.cpu().numpy().tolist()

        average_test_loss = test_loss / len(test_loader)
        test_accuracy = correct / total

        print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        context["ti"].xcom_push(key="predictions", value=predictions)

        return {
            "model_name": self.model_name,
            "timestamp": context["ts"],
            "average_test_loss": average_test_loss,
            "test_accuracy": test_accuracy,
        }
