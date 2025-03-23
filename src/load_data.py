

import deeplake
import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

def load_pacs_training_dataset():
  ds = deeplake.query('SELECT * FROM "hub://activeloop/pacs-train"')
  return ds.pytorch()

class VLCSImageDataset(Dataset):
    def __init__(self, root_dir, domain, dataset_split='full', transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        domain_path = os.path.join(root_dir, domain, dataset_split)
        label_folders = sorted(os.listdir(domain_path))

        for label_idx, label_folder in enumerate(label_folders):
            label_folder_path = os.path.join(domain_path, label_folder)
            if os.path.isdir(label_folder_path):
                for image_name in os.listdir(label_folder_path):
                    image_path = os.path.join(label_folder_path, image_name)
                    self.images.append(image_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {'images': image, 'labels': torch.tensor(label)}

def load_vlcs_dataset(domain, split='full', batch_size=64, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = VLCSImageDataset(
        root_dir=os.path.join(os.path.dirname(__file__), 'data', 'VLCS'),
        domain=domain,
        dataset_split=split,
        transform=transform
    )
    
    return dataset

