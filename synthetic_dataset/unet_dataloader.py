import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class SyntheticDataset(Dataset):
    def __init__(self, images_dir, targets_dir, targets_curved_dir, coeffs_dir, image_size, step):
        if step == 'extract':
            self.sources_dir = images_dir
            self.targets_dir = targets_curved_dir
        elif step == 'rectify':
            self.sources_dir = targets_curved_dir
            self.targets_dir = targets_dir
            self.coeffs_dir = coeffs_dir
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
        source_path = os.path.join(self.sources_dir, img_name)
        target_path = os.path.join(self.targets_dir, img_name)

        source = Image.open(source_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        if self.transform is not None:
            source = self.transform(source)
            target = self.transform(target)

        # Process the target image to extract the binary mask of the texts
        if self.step == 'extract':
            target = (target != 0).any(axis=0).float()

            return source, target
        else:
            source = (source != 0).any(axis=0).float().unsqueeze(0).repeat(3, 1, 1)
            target = (target != 0).any(axis=0).float()

            coeffs_path = os.path.join(self.coeffs_dir, img_name.replace('.png', '.npy'))
            coeffs = np.load(coeffs_path, allow_pickle=True).item()

            curvature_coefficients, perspective_coefficients = coeffs['curvature'], coeffs['perspective']
            curvature_coefficients = torch.as_tensor(curvature_coefficients, dtype=torch.float32)
            perspective_coefficients = torch.as_tensor(perspective_coefficients, dtype=torch.float32).flatten()
            coeffs = torch.cat((curvature_coefficients, perspective_coefficients), dim=0)

            return source, target, coeffs


