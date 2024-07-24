import os
from os.path import join, basename, relpath, splitext

from PIL import Image
from torch.utils.data import Dataset as TorchDataset

# Custom Dataset class for loading images and their labels
class CustomDataset(TorchDataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.samples = self.load_samples(data_dir)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        image = self.load_image(filename)
        image = self.transform(image)
        return image, label, filename

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(filename):
        # Load and convert an image to RGB
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    @staticmethod
    def load_samples(data_dir):
        # Load image paths and labels from the specified directory
        images = []
        labels = []

        cls_names = sorted([folder for folder in os.listdir(data_dir)])
        cls_to_idx = {cls_name: i for i, cls_name in enumerate(cls_names)}

        for root, _, filenames in os.walk(data_dir, topdown=False, followlinks=True):
            label = basename(relpath(root, data_dir)) if root != data_dir else ''
            for filename in filenames:
                base, ext = splitext(filename)
                if ext.lower() in ('.png', '.jpg', '.jpeg'):
                    images.append(join(root, filename))
                    labels.append(label)

        return [(i, cls_to_idx[j]) for i, j in zip(images, labels) if j in cls_to_idx]
