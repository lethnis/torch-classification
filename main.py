import os
import torch
from torch import optim, nn
from torchvision import transforms
from torchmetrics import Accuracy
from torchinfo import summary

from utils import ImageClassificationDataset, Trainer, Writer
from models import DenseNet, MobileNetv1


def main():
    """Main logic of a training and testing models."""

    # get hyperparameters
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 128
    EPOCHS = 5

    # get available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize writer will make a folder where training info will be saved
    writer = Writer("densenet")
    project_path = writer.project_path

    # initialize our model
    model = DenseNet([2, 2, 2], num_classes=10, is_bottleneck=True, k=8, start_channels=16)
    summary(model, (1, 3) + IMAGE_SIZE)

    # get transformations to prepare images
    transform = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE), transforms.TrivialAugmentWide(), transforms.ToTensor()]
    )

    # Note: for test set will be only applied Resize and ToTensor
    train_ds, test_ds = ImageClassificationDataset.from_full("../datasets/animals10", transform)
    # save some samples to the project directory we got from writer
    train_ds.save_samples(4, 4, os.path.join(project_path, "samples.png"))
    train_ds.save_transformed(4, 4, os.path.join(project_path, "transformed.png"))

    # initialize loss, accuracy and optimizer
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = Accuracy(task="multiclass", num_classes=10)
    optimizer = optim.Adam(model.parameters(), 0.001)

    # initialize trainer. It will make DataLoaders for train and test
    trainer = Trainer(train_ds, test_ds, BATCH_SIZE, device)

    # train the model. Additionally trainer will plot conf matrix and writer will record progress
    trainer.train(model, loss_fn, optimizer, accuracy_fn, writer, EPOCHS)

    # save trained model
    torch.save(model, os.path.join(project_path, "last.pt"))


if __name__ == "__main__":
    main()
