import random
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from transformers import set_seed
from datetime import datetime


id2label = {0: "background", 1: "lane"}

# map every class to a random color
id2color = {k: list(np.random.choice(range(256), size=3)) for k, v in id2label.items()}


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

train_transform = A.Compose([
    # hadded an issue with an image being too small to crop, PadIfNeeded didn't help...
    # if anyone knows why this is happening I'm happy to read why
    # A.PadIfNeeded(min_height=448, min_width=448),
    # A.RandomResizedCrop(height=448, width=448),
    A.Resize(width=448, height=448),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

test_transform = A.Compose([
    A.Resize(width=448, height=448),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])


def visualize_map(image, segmentation_map):
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)  # height, width, 3
    for label, color in id2color.items():
        color_seg[segmentation_map == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()


def seed_everything(seed: int = 42) -> None:
    if seed:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_cur_time() -> str:
    """
    return: 1970-01-01
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_cur_time_sec() -> str:
    """
    return: 1970-01-01 00:00:00
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

