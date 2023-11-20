import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        original_image = np.array(Image.open(item["image_path"]))
        original_segmentation_map = np.array(Image.open(item["label_path"]))
        original_segmentation_map = np.where(original_segmentation_map >= 1.0, 1.0, 0.0)

        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])

        # convert to C, H, W
        image = image.permute(2, 0, 1)

        return image, target, original_image, original_segmentation_map


def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[2] for i in inputs]
    batch["original_segmentation_maps"] = [i[3] for i in inputs]
    return batch
