import os
import sys
from pathlib import Path

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms


def main():
    """Main logic of predictions. Get the model, prepare images,
    predict on images and save predicted images.
    """
    model, images_paths, classes, img_size = parse_args()

    os.makedirs("predictions", exist_ok=True)

    # change images to the training size
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    for image_path in images_paths:
        # load image
        image = Image.open(image_path)
        # transform and add batch dimension
        transformed = transform(image).unsqueeze(0)
        # predict and get the highest score
        pred = classes[int(model(transformed).argmax(dim=1)[0])]

        # save image with class name
        plt.imshow(image)
        plt.title(pred)
        plt.axis("off")
        plt.savefig(f"predictions/{Path(image_path).name}")


def parse_args():

    MESSAGE = """Inference model on a single image or a folder with images.
    Usage: python predict.py path/to/best.pt path/to/images."""

    args = sys.argv[1:]

    # there should be exactly 2 arguments, model and images
    assert len(args) == 2, MESSAGE

    # extract arguments
    model_path = Path(args[0])
    images_paths = get_paths(args[1])

    # load model
    model = torch.load(model_path)

    # from model path extract classes and training args files
    classes_path = model_path.parent / "classes.txt"
    args_path = model_path.parent / "args.yaml"

    # store classes as a dict
    classes = {i: name for i, name in enumerate(classes_path.read_text().split("\n"))}

    # from training args extract image size
    img_size = args_path.read_text().split("\n")
    for line in img_size:
        if "img_size" in line:
            img_size = int(line.split(": ")[1])

    return model, images_paths, classes, img_size


def get_paths(base_dir):
    """Get all paths to files in base folder and all subfolders."""

    paths = []

    # if provided path is directory we search inside it
    if os.path.isdir(base_dir):
        # for every file and folder in base dir
        for i in os.listdir(base_dir):
            # update path so it will be full
            new_path = os.path.join(base_dir, i)
            # check if new path is a file or a folder
            if os.path.isdir(new_path):
                paths.extend(get_paths(new_path))
            else:
                paths.append(new_path)

    # Else path is a file, just add it to the resulting list
    else:
        paths.append(base_dir)

    return paths


if __name__ == "__main__":
    main()
