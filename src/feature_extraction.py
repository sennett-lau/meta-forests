import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

def custom_collate(batch):
    """Custom collate function that ensures tensors are writable"""
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, np.ndarray):
        # Make a writable copy of the NumPy array
        arr = np.stack([np.array(b) for b in batch])
        return torch.tensor(arr)
    else:
        return torch.utils.data._utils.collate.default_collate(batch)

def feature_extract_resnet(pacs_dataset, batch_size=64, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')

    # Load pre-trained ResNet-18 model with updated weights parameter
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])  # remove classification layer
    resnet18.eval().to(device)

    # Use custom collate function to handle non-writable arrays
    dataloader = DataLoader(pacs_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    features_list = []
    labels_list = []
    total_images = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            labels = batch['labels'].squeeze()  # Remove the extra dimension to get [B]
            
            # Convert images from [B, H, W, C] to [B, C, H, W]
            images = images.permute(0, 3, 1, 2)
            
            # Normalize to [0, 1] range
            images = images.float() / 255.0
            
            # Apply normalization (mean and std for ImageNet)
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            images = normalize(images)
            
            # Resize to 224x224 if needed (ResNet standard input size)
            if images.shape[2] != 224 or images.shape[3] != 224:
                resize = transforms.Resize((224, 224))
                images = resize(images)
            
            # Extract features
            features = resnet18(images).squeeze()
            
            # Handle case where batch size is 1
            if features.dim() == 1:
                features = features.unsqueeze(0)
                
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            total_images += images.shape[0]

    # Concatenate all features and labels
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    print(f'Processed {total_images} images')
    return features_array, labels_array
