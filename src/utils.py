from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from loss import Noise2VoidLoss
from model import NoiseToVoidUNet

import matplotlib.pyplot as plt

def normalize_image(np_img):
    if np_img.max() > 0:
        # Create mask of non-background pixels
        foreground_mask = np_img > 0.01
        if foreground_mask.any():
            # Get min/max of only foreground pixels
            fg_min = np_img[foreground_mask].min()
            fg_max = np_img[foreground_mask].max()
            
            # Normalize only foreground pixels to [0,1] range
            if fg_max > fg_min:
                np_img[foreground_mask] = (np_img[foreground_mask] - fg_min) / (fg_max - fg_min)
    
    # Force background to be true black
    np_img[np_img < 0.01] = 0
    return np_img

def normalize_data(data, target_min=0, target_max=1):
    """Normalize data to target range"""
    current_min = data.min()
    current_max = data.max()
    
    if current_min == current_max:
        return torch.ones_like(data) * target_min
    
    normalized = (data - current_min) / (current_max - current_min)
    normalized = normalized * (target_max - target_min) + target_min
    return normalized


def group_pairs(noisy_sets):
    noisy_pairs = []
    for noisy_group in noisy_sets:
        num_images = len(noisy_group)

        if num_images < 2:
            continue 

        j = 0
        while j < num_images - 1: 
            noisy_pairs.append((noisy_group[j], noisy_group[j + 1]))
            j += 2 

    print(f"Pairs: {len(noisy_pairs)}")

    return noisy_pairs

def get_unet_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_n2n_unet_model()
    model = model.to(device)

    criterion = Noise2VoidLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return device, model, criterion, optimizer

def get_stage1_loaders(full_noisy_pairs, img_size):

    noisy_pairs = group_pairs(full_noisy_pairs)
        
    transform = transforms.Compose([
        transforms.Lambda(lambda img: torch.tensor(img, dtype=torch.float32)), 
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    dataset_size = len(noisy_pairs)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    random.shuffle(noisy_pairs)

    train_data = noisy_pairs[:train_size]
    val_data = noisy_pairs[train_size:train_size + val_size]
    test_data = noisy_pairs[train_size + val_size:]

    train_dataset = Stage1(data=train_data, transform=transform)
    val_dataset = Stage1(data=val_data, transform=transform)
    test_dataset = Stage1(data=test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train size: {len(train_loader)}, Validation size: {len(val_loader)}, Test size: {len(test_loader)}")

    assert next(iter(train_loader))[0].shape == (1, 1, img_size, img_size), "First noisy image shape mismatch!"
    assert next(iter(train_loader))[1].shape == (1, 1, img_size, img_size), "Second noisy image shape mismatch!"

    return train_loader, val_loader, test_loader

def get_n2n_unet_model(in_channels=1, out_channels=1, device='cpu'):
    model = NoiseToVoidUNet(in_channels=in_channels, 
                           out_channels=out_channels,
                           features=[64, 128, 256, 512])
    return model.to(device)

def get_unet_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_n2n_unet_model()
    model = model.to(device)

    criterion = Noise2VoidLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return device, model, criterion, optimizer


# visualisation

def visualize_progress(model, input_tensor, target_tensor, epoch):
    """
    Visualize the current model's output during training
    
    Args:
        model: Current model
        input_tensor: Input tensor [1, 1, H, W]
        target_tensor: Target tensor [1, 1, H, W]
        epoch: Current epoch number
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get components
    input_np = input_tensor[0, 0].cpu().numpy()
    target_np = target_tensor[0, 0].cpu().numpy()
    flow_np = output['flow_component'][0, 0].cpu().numpy()
    noise_np = output['noise_component'][0, 0].cpu().numpy()
    denoised_np = input_np - noise_np
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original data
    # 
    axes[0, 0].imshow(input_np, cmap='gray')
    axes[0, 0].set_title("Input")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target_np, cmap='gray')
    axes[0, 1].set_title("Target")
    axes[0, 1].axis('off')
    
    # Row 2: Model outputs
    #normalise
    #axes[1, 0].imshow(flow_np, cmap='gray')
    flow_np = normalize_image(flow_np)
    axes[1, 0].imshow(flow_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("Flow Component")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(noise_np, cmap='gray')
    axes[1, 1].set_title("Noise Component")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(denoised_np, cmap='gray')
    axes[1, 2].set_title("Denoised (Input - Noise)")
    axes[1, 2].axis('off')
    
    plt.suptitle(f"Training Progress - Epoch {epoch}")
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot the training history
    
    Args:
        history: Dictionary containing loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['flow_loss'], label='Flow Loss')
    plt.plot(history['noise_loss'], label='Noise Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

class Stage1(Dataset):
    def __init__(self, data, transform=None):
        self.data = data 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_group = self.data[idx]

        img1, img2 = random.sample(noisy_group, 2)

        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1, axis=0)  # Add channel dim if grayscale
        if len(img2.shape) == 2:
            img2 = np.expand_dims(img2, axis=0)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        img1 = normalize_data(img1)
        img2 = normalize_data(img2)

        return img1, img2  # Return a tuple of two nois