"""
A convenience script to debug your model training configuration.
"""

from airflow.models.baseoperator import BaseOperator

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import ResNetForImageClassification
from PIL import Image


def transform_function(image):
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
        label_tensor = torch.tensor(label)

        image = Image.open(image_path)

        if self.transform_function:
            image = self.transform_function(image)

        # print("Image path: " + image_path, ". True label: " + str(label))
        # print("Shape transformed image: " + str(image.shape))

        return image, label_tensor


class TrainHuggingFaceBinaryImageClassifierOperator(BaseOperator):
    """
    Trains a binary HuggingFace image classification model on a list of locally saved images.

    :param model_name: name of the model to use as a string. Can reference a public HuggingFace model
    or be the path to a locally saved model.
    :param criterion: loss function.
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
        criterion,
        optimizer,
        local_images_filepaths: list,
        labels: list,
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

    def execute(self):
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

        print(f"Fetch model: {self.model_name}")

        model.classifier[-1] = torch.nn.Linear(
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print("Starting model training...")

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
                predictions = torch.round(torch.sigmoid(logits))

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



### SET YOUR PARAMETERS HERE ###

files = ["train_toy/" + x for x in os.listdir("train_toy/")]
labels = [
    0 if file_name.split("/")[-1].split(" ")[0] == "meningioma" else 1
    for file_name in files
]

train_classifier = TrainHuggingFaceBinaryImageClassifierOperator(
    task_id="debug_model",
    model_name="microsoft/resnet-50",
    criterion=torch.nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam,
    local_images_filepaths=files,
    labels=labels,
    learning_rate=0.05,
    model_save_dir="test_trainer/model_schwannoma_2",
    train_transform_function=transform_function,
    batch_size=10,
    num_epochs=2,
    shuffle=True,
)

# this executes the operator as if running in Airflow
train_classifier.execute()
