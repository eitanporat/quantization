import os
from torchvision import datasets
from .mapping import load_mapping
from pathlib import Path

class ImagenetDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        self.custom_mapping = load_mapping(Path(root) / 'mapping.json')
        super().__init__(root, transform=transform, target_transform=target_transform)

    def find_classes(self, directory):
        """Override find_classes to use the custom_mapping dictionary"""
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls: self.custom_mapping[cls] for cls in classes if cls in self.custom_mapping}
        return classes, class_to_idx
