import os
import sys
import torch
from torch import optim, nn
from torchvision import transforms
from torchmetrics import Accuracy
from torchinfo import summary

from utils import ImageClassificationDataset, Trainer, Writer
from models import DenseNet, MobileNetv1, _options


def main(args):
    """Main logic of a training and testing models."""

    # get available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get project name if it is available
    project_name = args["name"] if args["name"] is not None else args["data"]
    # initialize writer will make a folder where training info will be saved
    writer = Writer(project_name)
    args["project_path"] = writer.project_path
    print(f"Saving project to the {writer.project_path}")

    # save args as yaml file to the project path
    with open(os.path.join(args["project_path"], "args.yaml"), "w") as f:
        for k, v in args.items():
            f.write(f"{k}: {v}\n")

    # get transformations to prepare images
    transform = get_transforms(args["augs"], args["img_size"])

    # get train, val and test datasets if possible
    datasets = get_datasets(args["data"], transform)

    # initialize additional datasets with None, so we don't have any errors in training
    val_ds = None
    test_ds = None

    # initialize datasets
    if len(datasets) == 1:
        train_ds = next(iter(datasets))
    elif len(datasets) == 2:
        train_ds, val_ds = datasets
    elif len(datasets) == 3:
        train_ds, val_ds, test_ds = datasets
    else:
        print(f"Too many datasets in {args['data']}. Expected maximum 3 folders, got {len(os.listdir(args['data']))}")

    # save some samples to the project directory
    train_ds.save_samples(4, 4, os.path.join(args["project_path"], "samples.png"))
    train_ds.save_transformed(4, 4, os.path.join(args["project_path"], "transformed.png"))

    classes = train_ds.classes

    # save classes list as txt to the project folder
    open(os.path.join(args["project_path"], "classes.txt"), "w").write("\n".join(classes))

    # initialize our model
    model = get_model(args["model"], args["alpha"], args["reps"], args["bottleneck"], len(classes))
    summary(model, (1, 3) + (args["img_size"], args["img_size"]))

    # initialize loss, accuracy and optimizer
    loss_fn = nn.CrossEntropyLoss()
    if len(classes) == 2:
        accuracy_fn = Accuracy(task="binary")
    else:
        accuracy_fn = Accuracy(task="multiclass", num_classes=len(classes))
    optimizer = optim.Adam(model.parameters(), 0.001)

    # initialize trainer. It will make DataLoaders for train and test
    trainer = Trainer(train_ds, val_ds, test_ds, args["batch"], device)

    # train the model. Additionally trainer will plot conf matrix and writer will record progress
    trainer.train(model, loss_fn, optimizer, accuracy_fn, writer, args["epochs"])


def parse_args() -> dict:
    """Arguments parser from command line."""

    # usage message to prompt to user
    message = """
Usage: main.py path\\to\\dataset model [args]

Required args:

    data (str): path to folder with dataset. It may be folder with classes of folder
        with train-val-test splits with classes inside them.
        Example: 'path\\to\\dataset'.

    model (str): path to the model.pt file or one of available:
        1. mobilenetv1 - MobileNetv1 model. Use it with 'alpha' multiplier to control
            the size of the model.
            Examples: 'mobilenetv1' or 'mobilenetv1 alpha=0.5'.

        2. densenet - DenseNet model. Use it with reps list. For example reps=[4,4,4]
            means image will be looped through 4 DenseBlocks and then reduced by half in size.
            Then it will be looped and halved again. You can use one of premade models:
            'DenseNet121','DenseNet169','DenseNet201','DenseNet264'. Additionally you can
            specify whether to use bottleneck block styles or don't.
            Examples: 'densenet reps=[2,4,6,8,10] bottleneck=True' or 'densenet121'.

Optional args:

    alpha (float): multiplier of mobilenet. Better use small values like 0.5, 0.25.
        Example: 'alpha=0.25' Defaults to 1.0.

    reps (list(int)): list or repetitions for densenet model. Shows how many DenseBlocks
        will be used and how many times image will be resized by half.
        Examples: 'reps=[2,4]' or 'reps=[4,4,4,4,4]' Defaults to [2,4,8].
        
    bottleneck (bool): whether to use bottleneck style or don't in densenet.
        Example: 'bottleneck=True'. Defaults to False.
        
    batch (int): size of batches that will be fed into the model.
        Recommended to use powers of 2. E.g. 16, 32, 64, 128.
        Example: 'batch=32'. Defaults to 16.
    
    epochs (int): number of epochs to train the model.
        Example: 'epochs=100'. Defaults to 50.
        
    img_size (int): height and width of the image. Provide only one side.
        Example: 'img_size=320'. Defaults to 224.
        
    augs (str): augmentations/transformations that will be applied to training images.
        Needed to expand diversity of the data, prevent overfitting, increase
        generalization capability of the model.
        Available options are: 'no', 'soft', 'strong'.
        Example: 'augs='no'. Defaults to 'strong'.
        
    name (str): name of the project, where training info will be saved.
        Example: 'name=experiment1'. Defaults to the name of dataset."""

    # default args. If name is not provided it will be defined later in Writer class
    args = {
        "data": "",
        "model": "",
        "alpha": 1.0,
        "reps": [2, 4, 8],
        "bottleneck": False,
        "batch": 16,
        "epochs": 50,
        "img_size": 224,
        "augs": "strong",
        "name": None,
    }

    # first argument is main.py, remove it
    argv = sys.argv[1:]

    # user needs to provide at least path to dataset and model
    assert len(argv) >= 2, message + f"\n\nRequired at least 2 arguments. Provided {len(argv)}"

    # take path to data and model from terminal and store in args
    args["data"] = argv.pop(0)
    args["model"] = argv.pop(0)

    # split remaining by '=' sign. 'batch=16' -> {'batch':16}
    argv = {i.split("=")[0]: i.split("=")[1] for i in argv}

    # overwrite default args
    for k, v in argv.items():
        assert k in args.keys(), message + f"\n\nExpected values are {list(args.keys())[2:]}. Got '{k}'"
        # make bottleneck bool, not str
        if k == "bottleneck":
            args[k] = v == "True"
        # check if augs in allowed types
        elif k == "augs":
            assert v in ["no", "soft", "strong"], f"augs should be one of 'no', 'soft' or 'strong'. Got '{v}'"
            args[k] = v
        # update name if provided
        elif k == "name":
            args[k] = v
        # transform other types from str to their own types. '[2,4,6]' -> [2,4,6]
        else:
            args[k] = eval(v)

    return args


def get_transforms(augs: str, img_size: int) -> transforms.Compose:
    """Function generates transformations based on 'augs' argument.
    Important note: for val and test datasets will be only applied
    Resize and ToTensor.

    Args:
        augs (str): one of 'no', 'soft', 'strong' to decide what augmentations to use
        img_size (int): image size to resize

    Returns:
        transforms.Compose: list of transformations
    """

    if augs == "no":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

    elif augs == "soft":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.TrivialAugmentWide(1),
                transforms.ToTensor(),
                transforms.RandomErasing(scale=(0.05, 0.1)),
            ]
        )

    elif augs == "strong":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.TrivialAugmentWide(31),
                transforms.ToTensor(),
                transforms.RandomErasing(),
            ]
        )

    else:
        print(f"Augs value should be one of 'no', 'soft', 'strong'. Got {augs}")
        exit()


def get_datasets(path_to_dataset: str, transform: transforms.Compose) -> tuple:
    """Function gets datasets based on folder structure. If main folder contains
    subfolders like 'train', 'val', 'test' it will generate datasets from those
    folders. If in main folder straightaway classes it will take 90% for training
    and 10% for validating sets.

    Args:
        path_to_dataset (str): path to the main folder.
        transform (transforms.Compose): list of transforms from get_transforms function.

    Returns:
        list: list of datasets that may be only train, or train/val or train/val/test.
    """

    # try to find any of train, val or test
    folders = os.listdir(path_to_dataset)
    if any(i in folders for i in ["train", "test", "val", "valid", "validation"]):
        return ImageClassificationDataset.from_splitted(path_to_dataset, transform)

    return ImageClassificationDataset.from_full(path_to_dataset, transform)


def get_model(name_or_path: str, alpha: float, reps: list, bottleneck: bool, num_classes: int) -> torch.nn.Module:
    """Get the model to train. It could be predefined architecture or .pt file.
    Predefined are 'mobilenetv1' or 'densenet'. Read parse_args function for more info.

    Args:
        name_or_path (str): name of the model or file.
        alpha (float): multiplier for mobilenetv1 model.
        reps (list): number of repetitions for densenet.
        bottleneck (bool): whether to use bottleneck style or don't for densenet.
        num_classes (int): number of classes for final layer

    Returns:
        torch.nn.Module: PyTorch model
    """
    # use mobilenetv1 with alpha multiplier
    if name_or_path.lower() == "mobilenetv1":
        return MobileNetv1(3, num_classes=num_classes, alpha=alpha)
    # there is premade densenet architectures. Available densenet121/169/201/264
    elif name_or_path.lower() in [i.lower() for i in _options.Densenet_options.__args__]:
        return DenseNet.from_options(name_or_path, 3, num_classes=num_classes)
    # custom densenet
    elif name_or_path.lower() == "densenet":
        return DenseNet(reps, num_classes=num_classes, is_bottleneck=bottleneck)
    # loaded model
    else:
        return torch.load(name_or_path)


if __name__ == "__main__":
    main(parse_args())
