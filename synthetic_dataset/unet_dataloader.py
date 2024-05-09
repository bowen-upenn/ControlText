import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class SyntheticDataset(Dataset):
    def __init__(self, images_dir, targets_curved_dir, target_corners_dir, target_midlines_dir, image_size, step):
        if step == 'extract':
            self.sources_dir = images_dir
            self.targets_dir = targets_curved_dir
        elif step == 'rectify':
            self.sources_dir = targets_curved_dir
            self.target_corners_dir = target_corners_dir
            self.target_midlines_dir = target_midlines_dir
        else:
            raise ValueError('Invalid step. Please choose between "extract" and "rectify"')

        self.images_dir = images_dir
        self.images = os.listdir(images_dir)
        self.step = step
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((image_size, image_size), antialias=True)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        if step == 'extract':
            source_path = os.path.join(self.sources_dir, img_name)
            target_path = os.path.join(self.targets_dir, img_name)

            source = Image.open(source_path).convert('RGB')
            target = Image.open(target_path).convert('RGB')

            if self.transform is not None:
                source = self.transform(source)
                target = self.transform(target)

            # Process the target image to extract the binary mask of the texts
            target = (target != 0).any(axis=0).float()
            return source, target
        else:
            source_path = os.path.join(self.sources_dir, img_name)
            target_corners_path = os.path.join(self.target_corners_dir, img_name)
            target_midlines_path = os.path.join(self.target_midlines_dir, img_name)

            source = Image.open(source_path).convert('RGB')
            target_corners = Image.open(target_corners_path).convert('RGB')
            target_midlines = Image.open(target_midlines_path).convert('RGB')

            if self.transform is not None:
                source = self.transform(source)
                target_corners = self.transform(target_corners)
                target_midlines = self.transform(target_midlines)

            # Process the target image to extract the binary mask of the texts
            source = (source != 0).any(axis=0).float().unsqueeze(0).repeat(3, 1, 1)
            target_corners = (target_corners != 0).any(axis=0).float()
            target_midlines = (target_midlines != 0).any(axis=0).float()

            return source, target_corners, target_midlines
