import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from utils import normalize_image, get_stage1_loaders, get_unet_model, normalize_data
from models.enhanced_n2v_unet import get_e_unet_model

from ssm.ssm import SpeckleSeparationModule, SpeckleSeparationUNet


def ssim_loss(img1, img2, window_size=11, size_average=True):
    C1, C2 = 0.01**2, 0.03**2  # Small stability constants

    # Compute means
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    # SSIM calculation
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = num / den
    return 1 - ssim_map.mean() if size_average else 1 - ssim_map

# Total Variation (TV) Loss
def tv_loss(img):
    loss = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return loss

# Combined Loss Function
class Noise2VoidLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):
        super(Noise2VoidLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for SSIM loss
        self.gamma = gamma  # Weight for TV loss
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        ssim = ssim_loss(pred, target)
        tv = tv_loss(pred)
        return self.alpha * mse + self.beta * ssim + self.gamma * tv

def noise_smoothness_loss(noise_component, background_mask):
    """Encourage smoothness in the noise component"""
    # Extract noise in background regions (where mask = 1)
    background_noise = noise_component * background_mask
    
    # Calculate local variation within background regions
    # This penalizes high-frequency components in the noise
    diff_x = background_noise[:, :, :, 1:] - background_noise[:, :, :, :-1]
    diff_y = background_noise[:, :, 1:, :] - background_noise[:, :, :-1, :]
    
    # Sum of squared differences (L2 norm)
    smoothness_loss = torch.mean(diff_x**2) + torch.mean(diff_y**2)
    
    return smoothness_loss

def structure_separation_loss(noise_component, flow_component):
    """Ensure the noise component doesn't contain structural information"""
    # We want the noise and flow components to be uncorrelated
    # Calculate correlation coefficient between noise and flow
    noise_flat = noise_component.view(noise_component.size(0), -1)
    flow_flat = flow_component.view(flow_component.size(0), -1)
    
    # Normalize both components
    noise_norm = (noise_flat - torch.mean(noise_flat, dim=1, keepdim=True)) / torch.std(noise_flat, dim=1, keepdim=True)
    flow_norm = (flow_flat - torch.mean(flow_flat, dim=1, keepdim=True)) / torch.std(flow_flat, dim=1, keepdim=True)
    
    # Calculate cosine similarity (correlation)
    correlation = torch.sum(noise_norm * flow_norm, dim=1) / (noise_norm.size(1) - 1)
    
    # We want to minimize the absolute correlation
    return torch.mean(torch.abs(correlation))

def noise_distribution_regularization(noise_component, background_mask):
    """Encourage noise to follow expected statistical distribution"""
    # Extract background noise values
    bg_noise_values = noise_component[background_mask > 0.5]
    
    if bg_noise_values.numel() == 0:
        return torch.tensor(0.0, device=noise_component.device)
    
    # For speckle noise, often Rayleigh distribution is appropriate
    # Here we'll use a simple approach to match first and second moments
    
    # Calculate current statistics
    mean_noise = torch.mean(bg_noise_values)
    std_noise = torch.std(bg_noise_values)
    
    # Target statistics for noise (can be determined empirically)
    # For Rayleigh: mean ≈ σ√(π/2), std ≈ σ√(2-π/2)
    target_mean = 0.2  # Example value
    target_std = 0.15   # Example value
    
    # Penalize deviation from target statistics
    return torch.abs(mean_noise - target_mean) + torch.abs(std_noise - target_std)

def local_coherence_loss(noise_component, patch_size=7):
    """Encourage noise to have local coherence"""
    # Calculate local patch statistics
    unfold = nn.Unfold(kernel_size=patch_size, stride=1, padding=patch_size//2)
    patches = unfold(noise_component)
    patches = patches.view(noise_component.size(0), -1, noise_component.size(2), noise_component.size(3))
    
    # Calculate variance within each patch
    patch_mean = torch.mean(patches, dim=1, keepdim=True)
    patch_var = torch.mean((patches - patch_mean)**2, dim=1)
    
    # We want low variance within patches (locally smooth)
    return torch.mean(patch_var)

def structural_correlation_loss(noise_component, target):
    """
    Penalize correlation between the noise component and structural information in target
    """
    # We want the noise component to be uncorrelated with the target structures
    # Flatten the tensors for correlation calculation
    noise_flat = noise_component.view(noise_component.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Normalize to zero mean and unit variance for proper correlation measurement
    noise_norm = (noise_flat - torch.mean(noise_flat, dim=1, keepdim=True)) / (torch.std(noise_flat, dim=1, keepdim=True) + 1e-8)
    target_norm = (target_flat - torch.mean(target_flat, dim=1, keepdim=True)) / (torch.std(target_flat, dim=1, keepdim=True) + 1e-8)
    
    # Calculate correlation coefficient
    correlation = torch.sum(noise_norm * target_norm, dim=1) / (noise_norm.size(1) - 1)
    
    # Minimize the absolute correlation
    return torch.mean(torch.abs(correlation))

def custom_loss(flow_component, noise_component, batch_inputs, batch_targets):
    mse = nn.MSELoss(reduction='none')

    foreground_mask = (batch_targets > 0.03).float()
    background_mask = 1.0 - foreground_mask
    
    pixel_wise_loss = mse(flow_component, batch_targets)
    foreground_loss = (pixel_wise_loss * foreground_mask).sum() / (foreground_mask.sum() + 1e-8)
    
    background_loss = torch.mean((flow_component * background_mask)**2)

    edges_target = torch.abs(torch.nn.functional.conv2d(
        batch_targets, torch.tensor([[[[1, -1]]]], dtype=batch_targets.dtype, device=batch_targets.device)
    ))
    edges_flow = torch.abs(torch.nn.functional.conv2d(
        flow_component, torch.tensor([[[[1, -1]]]], dtype=flow_component.dtype, device=flow_component.device)
    ))
    edge_loss = mse(edges_flow, edges_target).mean()
    

    conservation_loss = nn.MSELoss()(batch_inputs, flow_component + noise_component)
    
    total_loss = (
        2.0 * foreground_loss +  # Prioritize vessel accuracy
        1.0 * background_loss +  # Strong penalty for background noise
        0.5 * edge_loss +         # Preserve vessel edges
        1.0 * conservation_loss  # Maintain physical consistency
    )
    
    return total_loss



def normalize_data(data, target_min=0, target_max=1):
    """Normalize data to target range"""
    current_min = data.min()
    current_max = data.max()
    
    if current_min == current_max:
        return torch.ones_like(data) * target_min
    
    normalized = (data - current_min) / (current_max - current_min)
    normalized = normalized * (target_max - target_min) + target_min
    return normalized

mse_loss = nn.MSELoss()

def threshold_octa(octa, method='adaptive', threshold_percentage=0.1):
    """
    Threshold OCTA image to remove low-intensity noise
    
    Args:
    - octa: Input OCTA image tensor
    - method: Thresholding method ('adaptive' or 'percentile')
    - threshold_percentage: Percentage for percentile-based thresholding
    
    Returns:
    - Thresholded OCTA image
    """
    # Ensure tensor is numpy or converted to numpy
    if torch.is_tensor(octa):
        octa = octa.detach().squeeze().cpu().numpy()
    
    if method == 'adaptive':
        # Otsu's method for adaptive thresholding
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(octa)
    elif method == 'percentile':
        # Percentile-based thresholding
        threshold = np.percentile(octa, (1 - threshold_percentage) * 100)
    else:
        # Default to percentile method
        threshold = np.percentile(octa, (1 - threshold_percentage) * 100)
    
    # Create binary mask
    #thresholded = (octa > threshold).astype(float)
    octa_np = octa
    thresholded = octa_np * (octa_np > threshold)
    
    # Convert back to tensor if needed
    if not torch.is_tensor(octa):
        thresholded = torch.from_numpy(thresholded).float()
    
    return thresholded.to('cuda')

def compute_decorrelation(oct1, oct2):

    # Full-spectrum decorrelation as used in Jia et al. paper
    numerator = (oct1 - oct2)**2
    denominator = oct1**2 + oct2**2
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    octa = numerator / (denominator + epsilon)

    #octa = threshold_octa(octa, method='percentile', threshold_percentage=0.01)
    
    return octa

def create_blindspot_mask(image, mask_ratio=0.1):
    """Creates blind spots in the image for N2V training"""
    #print(image.shape)
    batch_size, channels, height, width = image.shape
    masked_image = image.clone()
    
    # Create binary mask (1 for pixels to mask)
    mask = torch.zeros_like(image, dtype=torch.bool)
    
    # Randomly select pixels to mask (stratified sampling across image)
    target_indices = []
    num_pixels_to_mask = int(height * width * mask_ratio)
    
    for b in range(batch_size):
        indices = torch.randperm(height * width)[:num_pixels_to_mask]
        y_indices = indices // width
        x_indices = indices % width
        
        # Store indices for loss calculation
        target_indices.append((y_indices, x_indices))
        
        # Set mask
        for y, x in zip(y_indices, x_indices):
            mask[b, :, y, x] = True
            
            # Replace with value from neighborhood (e.g., pixel to the right)
            if x < width - 1:
                masked_image[b, :, y, x] = image[b, :, y, x+1]
            else:
                masked_image[b, :, y, x] = image[b, :, y, x-1]
    
    return masked_image, mask, target_indices

def masked_mse_loss(pred, target, target_indices):
    """Calculates MSE loss only at the masked positions"""
    loss = 0
    batch_size = pred.shape[0]
    
    for b in range(batch_size):
        y_indices, x_indices = target_indices[b]
        pred_values = pred[b, :, y_indices, x_indices]
        target_values = target[b, :, y_indices, x_indices]
        loss += torch.mean((pred_values - target_values)**2)
    
    return loss / batch_size

def visualise_stage2(oct1, oct2, denoised_oct1, denoised_oct2, octa_constraint, computed_octa):
    """
    Visualize the Stage 2 results using matplotlib.
    
    Args:
        oct1: First OCT B-scan (input)
        oct2: Second OCT B-scan (input)
        denoised_oct1: Denoised version of first OCT B-scan (output)
        denoised_oct2: Denoised version of second OCT B-scan (output)
        octa_constraint: Ground truth OCTA (from stage 1)
        computed_octa: OCTA computed from denoised OCT scans
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(oct1, torch.Tensor):
        oct1 = oct1.detach().cpu().squeeze().numpy()
    if isinstance(oct2, torch.Tensor):
        oct2 = oct2.detach().cpu().squeeze().numpy()
    if isinstance(denoised_oct1, torch.Tensor):
        denoised_oct1 = denoised_oct1.detach().cpu().squeeze().numpy()
    if isinstance(denoised_oct2, torch.Tensor):
        denoised_oct2 = denoised_oct2.detach().cpu().squeeze().numpy()
    if isinstance(octa_constraint, torch.Tensor):
        octa_constraint = octa_constraint.detach().cpu().squeeze().numpy()
    if isinstance(computed_octa, torch.Tensor):
        computed_octa = computed_octa.detach().cpu().squeeze().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot images
    axes[0, 0].imshow(oct1, cmap='gray')
    axes[0, 0].set_title('Input OCT 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(oct2, cmap='gray')
    axes[0, 1].set_title('Input OCT 2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(octa_constraint, cmap='gray')
    axes[0, 2].set_title('OCTA Constraint (Stage 1)')
    axes[0, 2].axis('off')
    
    #denoised_oct1_norm = normalize_image(denoised_oct1)
    axes[1, 0].imshow(denoised_oct1, cmap='gray')
    axes[1, 0].set_title('Denoised OCT 1')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(denoised_oct2, cmap='gray')
    axes[1, 1].set_title('Denoised OCT 2')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(computed_octa, cmap='gray')
    axes[1, 2].set_title('Computed OCTA')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualise_masks(oct1, oct2, masked_oct1, masked_oct2, mask1, mask2):

    # Convert tensors to numpy arrays if needed
    if isinstance(oct1, torch.Tensor):
        oct1 = oct1.detach().cpu().squeeze().numpy()
    if isinstance(oct2, torch.Tensor):
        oct2 = oct2.detach().cpu().squeeze().numpy()
    if isinstance(masked_oct1, torch.Tensor):
        masked_oct1 = masked_oct1.detach().cpu().squeeze().numpy()
    if isinstance(masked_oct2, torch.Tensor):
        masked_oct2 = masked_oct2.detach().cpu().squeeze().numpy()
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.detach().cpu().squeeze().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.detach().cpu().squeeze().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot images
    axes[0, 0].imshow(mask1, cmap='gray')
    axes[0, 0].set_title('Mask OCT 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask2, cmap='gray')
    axes[0, 1].set_title('Mask OCT 2')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(masked_oct1, cmap='gray')
    axes[1, 0].set_title('Masked OCT 1')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(masked_oct2, cmap='gray')
    axes[1, 1].set_title('Masked OCT 2')
    axes[1, 1].axis('off')



class OCTDenoiseDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, device):
        self.data = data_list
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        oct1, oct2, octa_constraint = self.data[idx]
        
        # Convert to torch tensors if they're numpy arrays
        if isinstance(oct1, np.ndarray):
            # Add single channel dimension - should be [C, H, W]
            oct1 = torch.tensor(oct1, dtype=torch.float32).unsqueeze(0)
        
        if isinstance(oct2, np.ndarray):
            # Add single channel dimension - should be [C, H, W]
            oct2 = torch.tensor(oct2, dtype=torch.float32).unsqueeze(0)
            
        if isinstance(octa_constraint, np.ndarray):
            octa_constraint = torch.tensor(octa_constraint, dtype=torch.float32)
            if octa_constraint.ndim == 3:  # If it already has a channel dimension
                pass
            else:
                # Add single channel dimension
                octa_constraint = octa_constraint.unsqueeze(0)
        
        # Ensure we have exactly 3 dimensions [C, H, W] before returning
        # DataLoader will add the batch dimension
        if oct1.ndim > 3:
            oct1 = oct1.squeeze()
            # Re-add channel dim if needed
            if oct1.ndim == 2:
                oct1 = oct1.unsqueeze(0)
                
        if oct2.ndim > 3:
            oct2 = oct2.squeeze()
            # Re-add channel dim if needed
            if oct2.ndim == 2:
                oct2 = oct2.unsqueeze(0)
                
        if octa_constraint.ndim > 3:
            octa_constraint = octa_constraint.squeeze()
            # Re-add channel dim if needed
            if octa_constraint.ndim == 2:
                octa_constraint = octa_constraint.unsqueeze(0)
        
        return oct1, oct2, octa_constraint

    def ___getitem__(self, idx):
        oct1, oct2, octa_constraint = self.data[idx]
        
        # Convert to torch tensors if they're numpy arrays
        if isinstance(oct1, np.ndarray):
            oct1 = torch.tensor(oct1, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        if isinstance(oct2, np.ndarray):
            oct2 = torch.tensor(oct2, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            
        if isinstance(octa_constraint, np.ndarray):
            octa_constraint = torch.tensor(octa_constraint, dtype=torch.float32)
            if octa_constraint.ndim == 3:  # If it already has a channel dimension
                pass
            else:
                octa_constraint = octa_constraint.unsqueeze(0)  # Add channel dimension
        
        return oct1, oct2, octa_constraint



def create_blindspot_mask(image, n_masks=64):
    # Check shape and print for debugging
    print(f"Image shape: {image.shape}")
    
    # Handle different dimension cases
    if len(image.shape) == 3:  # [channels, height, width]
        channels, height, width = image.shape
        batch_size = 1
        image = image.unsqueeze(0)  # Add batch dimension [1, channels, height, width]
    elif len(image.shape) == 4:  # [batch_size, channels, height, width]
        batch_size, channels, height, width = image.shape
    else:
        raise ValueError(f"Unexpected shape: {image.shape}")
    
    masked_image = image.clone()
    
    # For each image in the batch
    target_indices = []
    for b in range(batch_size):
        # Randomly select pixels to mask
        y_indices = torch.randint(0, height, (n_masks,))
        x_indices = torch.randint(0, width, (n_masks,))
        
        target_indices.append((y_indices, x_indices))
        
        # Create blind spots by replacing with neighboring pixels
        for i in range(n_masks):
            y, x = y_indices[i], x_indices[i]
            
            # Select a random neighbor
            neighbor_y = torch.clamp(y + torch.randint(-1, 2, (1,)), 0, height-1)
            neighbor_x = torch.clamp(x + torch.randint(-1, 2, (1,)), 0, width-1)
            
            # Replace target pixel with neighbor's value
            masked_image[b, :, y, x] = image[b, :, neighbor_y, neighbor_x]
    
    return masked_image, image, target_indices

import random

def create_blindspot_mask(image, n_masks=1024):
    """
    Create a blindspot mask for Noise2Void training.
    
    Args:
        image: Tensor of shape [C, H, W] or [B, C, H, W]
        n_masks: Number of blind spots to create
        
    Returns:
        masked_image: Image with blind spots
        original_image: Original image without modifications
        target_indices: Indices of masked pixels for loss calculation
    """

    if n_masks is None:
        _, _, height, width = image.shape
        n_masks = int(0.01 * height * width)

    # Convert to 4D tensor [B, C, H, W] if needed
    original_shape = image.shape
    
    if len(original_shape) == 3:  # [C, H, W]
        image = image.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
        
    # Get dimensions
    batch_size, channels, height, width = image.shape
    
    # Clone the image to create masked version
    masked_image = image.clone()
    
    # Initialize mask (1 where pixels are kept, 0 where masked)
    mask = torch.ones_like(image)
    
    # For each image in the batch
    all_target_indices = []
    
    for b in range(batch_size):
        # Randomly select pixels to mask
        y_indices = torch.randint(0, height, (n_masks,)).to(image.device)
        x_indices = torch.randint(0, width, (n_masks,)).to(image.device)
        
        # Store indices for loss calculation
        batch_indices = []
        for i in range(n_masks):
            y, x = y_indices[i], x_indices[i]
            batch_indices.append((y.item(), x.item()))
            
            # Create the mask (0 at masked positions)
            mask[b, :, y, x] = 0
            
            # Select a random neighbor
            offset_y = torch.randint(-1, 2, (1,)).item()
            offset_x = torch.randint(-1, 2, (1,)).item()
            
            # If offset is (0,0), choose a different one
            if offset_y == 0 and offset_x == 0:
                choices = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                offset_y, offset_x = random.choice(choices)
            
            # Calculate neighbor coordinates with boundary checks
            neighbor_y = max(0, min(height-1, y.item() + offset_y))
            neighbor_x = max(0, min(width-1, x.item() + offset_x))
            
            # Replace target pixel with neighbor's value
            masked_image[b, :, y, x] = image[b, :, neighbor_y, neighbor_x]
        
        all_target_indices.append(batch_indices)
    
    # Return to original shape if input was 3D
    if len(original_shape) == 3:
        masked_image = masked_image.squeeze(0)
        mask = mask.squeeze(0)

    print(f"Original vs masked difference: {torch.sum(torch.abs(image - masked_image)).item()}")
        
    return masked_image, mask, all_target_indices

def masked_mse_loss(predictions, targets, target_indices):
    batch_size = predictions.shape[0]
    total_loss = 0
    for b in range(batch_size):
        # Get all masked positions
        positions = target_indices[b]
        if not positions:
            continue
            
        # Extract predictions and targets at masked positions
        pred_vals = torch.stack([predictions[b, :, y, x] for y, x in positions])
        target_vals = torch.stack([targets[b, :, y, x] for y, x in positions])
        
        # Calculate MSE
        batch_loss = F.mse_loss(pred_vals, target_vals)
        total_loss += batch_loss
    
    return total_loss / batch_size

def sophisticated_loss(denoised_oct1, denoised_oct2, oct1, oct2, octa_constraint, target_indices1, target_indices2):
    # Calculate individual losses
    n2v_loss1 = masked_mse_loss(denoised_oct1, oct1, target_indices1)
    n2v_loss2 = masked_mse_loss(denoised_oct2, oct2, target_indices2)
    n2v_loss = n2v_loss1 + n2v_loss2
    
    computed_octa = compute_decorrelation(denoised_oct1, denoised_oct2)
    
    threshold = 0.05
    vessel_mask = (octa_constraint > threshold).float()
    non_vessel_mask = 1.0 - vessel_mask
    
    structure_preservation = F.mse_loss(denoised_oct1 * vessel_mask, oct1 * vessel_mask)
    structure_loss = F.mse_loss(computed_octa * vessel_mask, octa_constraint * vessel_mask)
    background_loss = torch.mean((computed_octa * non_vessel_mask)**2)
    
    # Gradient components for edge preservation
    grad_x_denoised = denoised_oct1[:,:,1:,:] - denoised_oct1[:,:,:-1,:]
    grad_y_denoised = denoised_oct1[:,:,:,1:] - denoised_oct1[:,:,:,:-1]
    grad_x_orig = oct1[:,:,1:,:] - oct1[:,:,:-1,:]
    grad_y_orig = oct1[:,:,:,1:] - oct1[:,:,:,:-1]
    gradient_loss = F.l1_loss(grad_x_denoised, grad_x_orig) + F.l1_loss(grad_y_denoised, grad_y_orig)
    
    # Balance components - key adjustments here
    n2v_loss = 1.0 * n2v_loss
    structure_loss = 0.0 * structure_loss
    background_loss = 0.0 * background_loss
    structure_preservation = 0.0 * structure_preservation
    gradient_loss = 0.0 * gradient_loss
    correlation_loss = 0.0 * (1.0 - F.cosine_similarity(computed_octa.flatten(), octa_constraint.flatten(), dim=0))

    total_loss = n2v_loss + structure_loss + background_loss + structure_preservation + gradient_loss + correlation_loss
    
    # Log all components to see their values
    component_values = {
        "n2v_loss": n2v_loss.item(),
        "structure_loss": structure_loss.item(),
        "background_loss": background_loss.item(),
        "structure_preservation": structure_preservation.item(),
        "gradient_loss": gradient_loss.item(),
        "total_loss": total_loss.item()
    }
    
    print(f"Loss components: {component_values}")
    
    return total_loss, component_values

def normalize_intensity(denoised, reference):
    """
    Normalizes the intensity of denoised OCT images to match the reference images.
    This helps maintain similar intensity distribution after denoising.
    
    Args:
        denoised: Tensor of denoised images [B, C, H, W]
        reference: Tensor of reference images [B, C, H, W]
        
    Returns:
        Normalized denoised image with intensity distribution matching reference
    """
    # Calculate mean and std for each image in batch
    batch_size = denoised.shape[0]
    normalized = denoised.clone()
    
    for b in range(batch_size):
        # Get mean and std of each image
        d_mean = torch.mean(denoised[b])
        d_std = torch.std(denoised[b])
        r_mean = torch.mean(reference[b])
        r_std = torch.std(reference[b])
        
        # Normalize: (x - mean) / std * target_std + target_mean
        normalized[b] = (denoised[b] - d_mean) / (d_std + 1e-6) * r_std + r_mean
    
    return normalized

def ssn2v_loss(denoised_oct1, denoised_oct2, oct1, oct2, octa_constraint, target_indices1, target_indices2, computed_octa):
    # Step 1: N2V denoising loss (blind-spot training)
    n2v_loss1 = masked_mse_loss(denoised_oct1, oct1, target_indices1)
    n2v_loss2 = masked_mse_loss(denoised_oct2, oct2, target_indices2)
    denoising_loss = n2v_loss1 + n2v_loss2
    
    # Step 2: OCTA flow information preservation constraint
    
    octa_loss = F.mse_loss(computed_octa, octa_constraint)
    #octa_loss = gradient_aware_octa_loss(computed_octa, octa_constraint)

    intensity_loss = histogram_loss(denoised_oct1, oct1) + histogram_loss(denoised_oct2, oct2)

    
    # Combine losses - emphasize OCTA preservation
    alpha = 5  # Weight for OCTA constraint (increase to preserve more flow information)
    beta = 10
    total_loss = denoising_loss + alpha * octa_loss + intensity_loss * beta

    print(f"Loss components: {denoising_loss.item()}, {octa_loss.item()}, {intensity_loss.item()}")
    print(f"Total loss: {total_loss.item()}")
    
    return total_loss, {"denoising_loss": denoising_loss.item(), "octa_loss": octa_loss.item()}

def _ssn2v_loss(denoised_oct1, denoised_oct2, oct1, oct2, octa_constraint, target_indices1, target_indices2, computed_octa):
    # Step 1: N2V denoising loss (blind-spot training)
    n2v_loss1 = masked_mse_loss(denoised_oct1, oct1, target_indices1)
    n2v_loss2 = masked_mse_loss(denoised_oct2, oct2, target_indices2)
    denoising_loss = n2v_loss1 + n2v_loss2
    
    # Step 2: OCTA flow information preservation constraint
    #computed_octa = compute_decorrelation(denoised_oct1, denoised_oct2)
    #computed_octa = ssm(denoised_oct1.to('cuda'))['flow_component']
    
    # Simplify the OCTA loss - use basic MSE without gradients to reduce magnitude
    #plt.imshow(computed_octa, cmap='gray')
    #plt.imshow(octa_constraint, cmap='gray')
    octa_loss = F.mse_loss(computed_octa, octa_constraint)
    
    # Or retain gradient awareness but scale components individually
    # grad_x_loss = F.mse_loss(computed_grad_x, target_grad_x)
    # grad_y_loss = F.mse_loss(computed_grad_y, target_grad_y)
    # octa_loss = octa_mse + 0.1 * (grad_x_loss + grad_y_loss)  # Reduce gradient contribution
    
    intensity_loss = histogram_loss(denoised_oct1, oct1) + histogram_loss(denoised_oct2, oct2)
    
    # Apply log-scaling to reduce the impact of magnitude differences
    log_denoising = torch.log1p(denoising_loss)
    log_octa = torch.log1p(octa_loss)
    log_intensity = torch.log1p(intensity_loss)
    
    alpha = 1.0  
    beta = 1.0
    
    total_loss = log_denoising + alpha * log_octa + beta * log_intensity

    print(f"Loss components: {denoising_loss.item()}, {octa_loss.item()}, {intensity_loss.item()}")
    print(f"Total loss: {total_loss.item()}")
    
    return total_loss, {"denoising_loss": denoising_loss.item(), "octa_loss": octa_loss.item()}

def histogram_loss(denoised, original):
    return F.mse_loss(torch.sort(denoised.flatten())[0], torch.sort(original.flatten())[0])

def gradient_aware_octa_loss(computed_octa, octa_constraint):
    # Calculate MSE
    mse_loss = F.mse_loss(computed_octa, octa_constraint)
    
    # Calculate gradient preservation loss
    computed_grad_x = computed_octa[:, :, :, 1:] - computed_octa[:, :, :, :-1]
    computed_grad_y = computed_octa[:, :, 1:, :] - computed_octa[:, :, :-1, :]
    
    target_grad_x = octa_constraint[:, :, :, 1:] - octa_constraint[:, :, :, :-1]
    target_grad_y = octa_constraint[:, :, 1:, :] - octa_constraint[:, :, :-1, :]
    
    grad_loss_x = F.mse_loss(computed_grad_x, target_grad_x)
    grad_loss_y = F.mse_loss(computed_grad_y, target_grad_y)
    
    # Combined loss
    return mse_loss + 1 * (grad_loss_x + grad_loss_y)

def train_stage2(model, device, optimizer, train_loader, val_loader, num_epochs=5, scheduler=None, visualise=False):

    ssm_path = "checkpoints/speckle_separation_model.pth"
    ssm = SpeckleSeparationModule(input_channels=1, feature_dim=32)
    ssm.load_state_dict(torch.load(ssm_path, map_location=device))
    ssm.to(device)
    ssm.eval()

    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (oct1, oct2, octa_constraint) in enumerate(train_loader):
            # Move to device
            oct1, oct2, octa_constraint = oct1.to(device), oct2.to(device), octa_constraint.to(device)

            #print(f"Batch shapes - OCT1: {oct1.shape}, OCT2: {oct2.shape}, OCTA: {octa_constraint.shape}")
            
            # Create masked versions of OCT images for N2V training
            masked_oct1, mask1, target_indices1 = create_blindspot_mask(oct1)
            masked_oct2, mask2, target_indices2 = create_blindspot_mask(oct2)
            
            # Denoise each OCT image
            try:
                denoised_oct1, _ = model(masked_oct1)
                denoised_oct2, _ = model(masked_oct2)
            except:
                denoised_oct1 = model(masked_oct1)
                denoised_oct2 = model(masked_oct2)

            denoised_oct1 = normalize_intensity(denoised_oct1, oct1)
            denoised_oct2 = normalize_intensity(denoised_oct2, oct2)
            

            computed_octa = compute_decorrelation(denoised_oct1, denoised_oct2)
            plt.imshow(computed_octa[0][0].detach().cpu().numpy(), cmap='gray')

            #computed_octa = ssm(computed_octa)['flow_component']
            #computed_octa_np = computed_octa.detach().cpu().numpy()
            #normalized_octa = normalize_image(computed_octa_np)
            #octa_sample = normalized_octa[0][0] 
            #computed_octa_tensor = torch.tensor(octa_sample, device=device)
            #computed_octa = normalize_image(computed_octa.detach().cpu().numpy())[0][0].to(device)
            #print(f"Masked pixels: {len(target_indices1[0])}, Unique positions: {len(set((y,x) for y,x in target_indices1[0]))}")

            total_loss, loss_components = ssn2v_loss(
                denoised_oct1, denoised_oct2, 
                oct1, oct2, 
                octa_constraint,
                target_indices1, target_indices2,
                computed_octa
            )
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if visualise and epoch % 2 == 0:
                print(f"Batch {batch_idx}, Loss: {total_loss.item():.4f}")
                from IPython.display import clear_output
                clear_output()
                visualise_stage2(oct1[0], oct2[0], denoised_oct1[0], denoised_oct2[0], 
                                octa_constraint[0], computed_octa)
                
                
        epoch_loss += total_loss.item()

        val_loss, val_psnr = validate(model, device, val_loader, ssm)

        epoch_loss /= len(train_loader)  # Divide by number of batches

        scheduler.step(val_loss)

import torch

def apply_model(model, oct1, oct2, device):

    oct1 = torch.from_numpy(oct1) if isinstance(oct1, np.ndarray) else oct1
    oct2 = torch.from_numpy(oct2) if isinstance(oct2, np.ndarray) else oct2

    # Add batch dimension if not already present
    if oct1.ndim == 3:  # Assuming [C, H, W] format
        oct1 = oct1.unsqueeze(0)  # -> [1, C, H, W]
    if oct2.ndim == 3:
        oct2 = oct2.unsqueeze(0)

    # Move to device
    oct1, oct2 = oct1.to(device), oct2.to(device)

    masked_oct1, mask1, target_indices1 = create_blindspot_mask(oct1)
    masked_oct2, mask2, target_indices2 = create_blindspot_mask(oct2)

    try:
        denoised_oct1, _ = model(masked_oct1)
        denoised_oct2, _ = model(masked_oct2)
    except:
        denoised_oct1 = model(masked_oct1)
        denoised_oct2 = model(masked_oct2)

    denoised_oct1 = normalize_intensity(denoised_oct1, oct1)
    denoised_oct2 = normalize_intensity(denoised_oct2, oct2)

    return denoised_oct1, denoised_oct2




def _process_stage2(results, model, history, train_loader, val_loader, test_loader, epochs=1, stage1_path='checkpoints/stage1.pth', visualise=False):


    stage2_data = []
    n = 20  # Number of samples to process
    for i in range(n):

        oct1 = results['raw_image'][i]
        oct2 = results['raw_image'][i+1]  
        

        denoised_octa = results['flow_components'][i]
        
        stage2_data.append((oct1, oct2, denoised_octa))

    print("Stage 2 data length: ", len(stage2_data))


    #device, model, criterion, optimizer = get_unet_model()
    device, model, criterion, optimizer = get_e_unet_model()

    checkpoint = torch.load(stage1_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded")

    mse_loss = nn.MSELoss()
    '''
    dataset = OCTDenoiseDataset(stage2_data, device)
    batch_size = 1  # Adjust as needed
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)'''

    from torch.utils.data import DataLoader, random_split

    # Create the full dataset
    dataset = OCTDenoiseDataset(stage2_data, device)

    # Define the size of each split
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)  # 70% for training
    val_size = int(0.15 * dataset_size)   # 15% for validation
    test_size = dataset_size - train_size - val_size  # Remaining for testing

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create data loaders for each split
    batch_size = 1  # Adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train size: {len(train_loader.dataset)}, Validation size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")
    assert len(train_loader) > 0, "Train loader is empty!"
    assert len(val_loader) > 0, "Validation loader is empty!"

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    model2 = model

    for name, param in model2.named_parameters():
        if 'down' in name:  # Only train the decoder part
            param.requires_grad = False

    train_stage2(model2, device, optimizer, train_loader, val_loader, num_epochs=epochs, scheduler=scheduler, visualise=visualise)

    save_path = r"checkpoints/stage2_256_final_N2NUNet_model.pth"

    torch.save({'model_state_dict': model.state_dict(),}, save_path)

    denoised_oct1, denoised_oct2 = apply_model(model2, oct1, oct2, device)
    return denoised_oct1, denoised_oct2

def validate(model, device, data_loader, ssm):
    """
    Validate the model on a validation dataset.
    
    Args:
        model: The neural network model
        device: The device to run validation on (CPU or GPU)
        data_loader: DataLoader for validation data
        
    Returns:
        avg_loss: Average loss over the validation set
        avg_psnr: Average PSNR over the validation set
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_psnr = 0.0
    
    with torch.no_grad():  # No need to track gradients
        for batch_idx, (oct1, oct2, octa_constraint) in enumerate(data_loader):
            # Move to device
            oct1, oct2, octa_constraint = oct1.to(device), oct2.to(device), octa_constraint.to(device)
            
            # Create masked versions of OCT images for N2V training
            masked_oct1, mask1, target_indices1 = create_blindspot_mask(oct1)
            masked_oct2, mask2, target_indices2 = create_blindspot_mask(oct2)
            
            # Denoise each OCT image
            try:
                denoised_oct1, _ = model(masked_oct1)
                denoised_oct2, _ = model(masked_oct2)
            except:
                denoised_oct1 = model(masked_oct1)
                denoised_oct2 = model(masked_oct2)
            
            denoised_oct1 = normalize_intensity(denoised_oct1, oct1)
            denoised_oct2 = normalize_intensity(denoised_oct2, oct2)
            
            computed_octa = compute_decorrelation(denoised_oct1, denoised_oct2)
            #computed_octa = ssm(denoised_oct1.to('cuda'))['flow_component']
            #computed_octa_np = computed_octa.detach().cpu().numpy()
            #normalized_octa = normalize_image(computed_octa_np)
            #octa_sample = normalized_octa[0][0] 
            #computed_octa_tensor = torch.tensor(octa_sample, device=device)
            
            # Calculate loss
            loss, _ = ssn2v_loss(
                denoised_oct1, denoised_oct2, 
                oct1, oct2, 
                octa_constraint,
                target_indices1, target_indices2,
                computed_octa
            )
            
            # Calculate PSNR
            mse1 = torch.mean((denoised_oct1 - oct1) ** 2)
            mse2 = torch.mean((denoised_oct2 - oct2) ** 2)
            avg_mse = (mse1 + mse2) / 2
            psnr = 10 * torch.log10(1.0 / avg_mse)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
    
    # Calculate averages
    avg_loss = total_loss / len(data_loader)
    avg_psnr = total_psnr / len(data_loader)
    
    return avg_loss, avg_psnr

# import random split
from torch.utils.data import random_split

def process_stage2(results, model, history, train_loader, val_loader, test_loader, epochs=1, stage1_path='checkpoints/stage1.pth', visualise=False, load=False):
    # Prepare stage 2 data
    stage2_data = []
    n = min(10, len(results['raw_image']) - 1)  # Ensure we don't go out of bounds
    for i in range(n):
        oct1 = results['raw_image'][i]
        oct2 = results['raw_image'][i+1]  
        denoised_octa = results['flow_components'][i]
        
        stage2_data.append((oct1, oct2, denoised_octa))

    print(f"Stage 2 data length: {len(stage2_data)}")

    # Device and model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, model, _, optimizer = get_e_unet_model()

    # Load stage 1 checkpoint
    if load:
        checkpoint = torch.load(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\src\checkpoints\stage2_256_final_N2NUNet_model_backup.pth", map_location=device)
    else:
        checkpoint = torch.load(stage1_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Stage 1 model loaded")

    # Create dataset and data loaders
    dataset = OCTDenoiseDataset(stage2_data, device)
    dataset_size = len(dataset)
    
    # Precise data splitting
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Data loader configuration
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Logging dataset sizes
    print(f"Train size: {len(train_loader.dataset)}, "
          f"Validation size: {len(val_loader.dataset)}, "
          f"Test size: {len(test_loader.dataset)}")
    
    # Sanity checks
    assert len(train_loader) > 0, "Train loader is empty!"
    assert len(val_loader) > 0, "Validation loader is empty!"

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Partial model freezing (optional)
    for name, param in model.named_parameters():
        if 'down' in name:  # Only train the decoder part
            param.requires_grad = False

    # Stage 2 training
    model = train_stage2(
        model, 
        device, 
        optimizer, 
        train_loader, 
        val_loader, 
        num_epochs=epochs, 
        scheduler=scheduler, 
        visualise=visualise
    )

    # Save model
    save_path = "checkpoints/stage2_256_final_N2NUNet_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)

    # Apply model to sample images
    denoised_oct1, denoised_oct2 = apply_model(model, oct1, oct2, device)
    
    return denoised_oct1, denoised_oct2

###


class Noise2VoidLoss(nn.Module):
    def __init__(self, mask_ratio=0.1):
        super(Noise2VoidLoss, self).__init__()
        self.mask_ratio = mask_ratio
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, prediction, target):
        # Create a binary mask where 1 indicates masked pixels
        mask = torch.bernoulli(torch.full_like(target, self.mask_ratio)).bool()
        
        # Compute pixel-wise MSE loss
        pixel_losses = self.mse(prediction, target)
        
        # Only compute loss for masked pixels
        masked_losses = pixel_losses[mask]
        
        # Avoid division by zero
        if masked_losses.numel() == 0:
            return torch.tensor(0.0, device=prediction.device)
        
        # Compute mean loss over masked pixels
        loss = masked_losses.mean()
        
        return loss

def train_stage2(model, device, optimizer, train_loader, val_loader, num_epochs=5, scheduler=None, visualise=False):

    ssm_path = "checkpoints/speckle_separation_model.pth"
    ssm = SpeckleSeparationModule(input_channels=1, feature_dim=32)
    #ssm = SpeckleSeparationUNet(input_channels=1, feature_dim=32)
    ssm.load_state_dict(torch.load(ssm_path, map_location=device))
    ssm.to(device)
    ssm.eval()

    
    
    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        
        for batch_idx, (oct1, oct2, octa_constraint) in enumerate(train_loader):
            print("Training", "Epoch: ", epoch, "Batch: ", batch_idx)
            # Move to device
            oct1, oct2, octa_constraint = oct1.to(device), oct2.to(device), octa_constraint.to(device)
            
            # Create masked versions using N2V approach
            masked_oct1, mask1, target_indices1 = create_blindspot_mask_fast(oct1)
            masked_oct2, mask2, target_indices2 = create_blindspot_mask_fast(oct2)
            
            
            # Denoise images
            denoised_oct1 = model(masked_oct1)
            denoised_oct2 = model(masked_oct2)
            
            # Compute OCTA 
            computed_octa = compute_decorrelation(denoised_oct1, denoised_oct2)
            #computed_octa = ssm(denoised_oct1)['flow_component']
            
            
            # Compute loss using the custom loss function
            total_loss = compute_stage2_loss(
                denoised_oct1, denoised_oct2, 
                oct1, oct2, 
                octa_constraint, 
                computed_octa,
                target_indices1,
                target_indices2
            )
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            # Optional visualization
            if visualise:
                # clear
                from IPython.display import clear_output
                clear_output(wait=True)
                visualise_stage2(
                    oct1, oct2, 
                    denoised_oct1, denoised_oct2, 
                    octa_constraint, 
                    computed_octa
                )
        
        # Validation
        val_loss = validate_stage2(model, val_loader)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
    
    return model

def compute_n2v_loss(denoised, original, target_indices):
    """
    Compute Noise2Void loss for specific masked indices
    
    Args:
    - denoised: Denoised image tensor
    - original: Original image tensor
    - target_indices: Indices of masked pixels
    
    Returns:
    - Loss value for masked pixels
    """
    # Ensure tensors are on the same device
    denoised = denoised.to(original.device)
    
    # Check if target_indices is a tuple or list (from create_blindspot_mask)
    if isinstance(target_indices, tuple) or isinstance(target_indices, list):
        # Assuming target_indices is a tuple of (batch_indices, row_indices, col_indices)
        batch_indices, row_indices, col_indices = target_indices
        
        # Create a mask tensor
        mask = torch.zeros_like(original, dtype=torch.bool)
        mask[batch_indices, 0, row_indices, col_indices] = True
        
        # Extract masked values
        denoised_masked = denoised[mask]
        original_masked = original[mask]
    else:
        # If target_indices is already a mask
        mask = target_indices
        denoised_masked = denoised[mask]
        original_masked = original[mask]
    
    # Compute MSE loss for masked pixels
    if denoised_masked.numel() > 0:
        n2v_loss = F.mse_loss(denoised_masked, original_masked)
    else:
        n2v_loss = torch.tensor(0.0, device=original.device)
    
    return n2v_loss

def validate_stage2(model, val_loader):
    """
    Validate the model on validation dataset
    
    Args:
    - model: Trained model
    - val_loader: Validation data loader
    
    Returns:
    - Average validation loss
    """
    model.eval()
    total_val_loss = 0.0

    device = 'cuda'
    
    with torch.no_grad():
        for oct1, oct2, octa_constraint in val_loader:

            print("Validating")

            oct1 = oct1.to(device)
            oct2 = oct2.to(device)
            octa_constraint = octa_constraint.to(device)
            # Perform validation similar to training
            masked_oct1, _, target_indices1 = create_blindspot_mask(oct1)
            masked_oct2, _, target_indices2 = create_blindspot_mask(oct2)
            
            denoised_oct1 = model(masked_oct1)
            denoised_oct2 = model(masked_oct2)
            
            computed_octa = compute_decorrelation(denoised_oct1, denoised_oct2)
            
            val_loss = compute_stage2_loss(
                denoised_oct1, denoised_oct2, 
                oct1, oct2, 
                octa_constraint, 
                computed_octa,
                target_indices1,
                target_indices2
            )
            
            total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

def compute_n2v_loss(denoised, original, target_indices):
    """
    Compute Noise2Void loss for specific masked indices
    
    Args:
    - denoised: Denoised image tensor
    - original: Original image tensor
    - target_indices: Tuple of indices where mask was applied
    
    Returns:
    - Loss value for masked pixels
    """
    # Unpack target indices
    batch_indices, row_indices, col_indices = target_indices
    
    # Extract masked pixel values
    denoised_masked = denoised[batch_indices, 0, row_indices, col_indices]
    original_masked = original[batch_indices, 0, row_indices, col_indices]
    
    # Compute MSE loss for masked pixels
    if denoised_masked.numel() > 0:
        n2v_loss = F.mse_loss(denoised_masked, original_masked)
    else:
        n2v_loss = torch.tensor(0.0, device=original.device)
    
    return n2v_loss

def compute_stage2_loss(denoised_oct1, denoised_oct2, 
                         original_oct1, original_oct2, 
                         octa_constraint, 
                         computed_octa,
                         target_indices1,
                         target_indices2):
    """
    Comprehensive loss computation for stage 2 training
    """
    # Ensure tensors are on the same device
    device = original_oct1.device
    
    # Normalize tensors to prevent extreme values
    denoised_oct1 = normalize_tensor(denoised_oct1)
    denoised_oct2 = normalize_tensor(denoised_oct2)
    original_oct1 = normalize_tensor(original_oct1)
    original_oct2 = normalize_tensor(original_oct2)
    octa_constraint = normalize_tensor(octa_constraint)
    computed_octa = normalize_tensor(computed_octa)
    
    # N2V Loss with gradient clipping
    n2v_loss = compute_n2v_loss(denoised_oct1, original_oct1, target_indices1)
    n2v_loss += compute_n2v_loss(denoised_oct2, original_oct2, target_indices2)
    
    # OCTA Constraint Loss with reduction
    octa_loss = F.mse_loss(computed_octa, octa_constraint, reduction='mean')
    
    # Denoising Quality Loss with regularization
    denoising_loss = F.mse_loss(denoised_oct1, original_oct1) + F.mse_loss(denoised_oct2, original_oct2)

    #diversity_loss = torch.std(denoised)

    feature_loss = torch.nn.functional.l1_loss(denoised_oct1, original_oct1) + torch.nn.functional.l1_loss(denoised_oct2, original_oct2)

    alpha, beta, gamma, delta = 0,0,0,0

    #alpha = 0.05
    beta = 0.5
    gamma = 0.25
    delta = 0.25

    # Loss weighting
    total_loss = (
        alpha * n2v_loss + 
        beta * octa_loss + 
        gamma * denoising_loss +
        delta * feature_loss
    )

    print(f"Loss components: N2V: {n2v_loss.item()}, OCTA: {octa_loss.item()}, Denoising: {denoising_loss.item()}, Feature: {feature_loss.item()}")
    print(f"Total loss: {total_loss.item()}")
    
    return total_loss

def normalize_tensor(tensor, min_val=0, max_val=1):
    """
    Normalize tensor to a specified range
    
    Args:
    - tensor: Input tensor
    - min_val: Minimum value after normalization
    - max_val: Maximum value after normalization
    
    Returns:
    - Normalized tensor
    """
    tensor = tensor.clone()
    
    # Handle single-channel images
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    
    # Compute min and max for each image in the batch
    batch_min = tensor.view(tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    batch_max = tensor.view(tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    
    # Normalize to [0, 1]
    tensor = (tensor - batch_min) / (batch_max - batch_min + 1e-8)
    
    # Scale to desired range
    tensor = tensor * (max_val - min_val) + min_val
    
    return tensor

def create_blindspot_mask(tensor, mask_ratio=0.1):
    """
    Create a blindspot mask for Noise2Void training
    
    Args:
    - tensor: Input tensor
    - mask_ratio: Proportion of pixels to mask
    
    Returns:
    - masked_tensor: Tensor with some pixels replaced
    - mask: Boolean mask of masked pixels
    - target_indices: Indices of masked pixels
    """
    # Ensure tensor is 4D (B, C, H, W)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    B, C, H, W = tensor.shape
    device = tensor.device
    
    # Create a mask
    mask = torch.rand(B, C, H, W, device=device) < mask_ratio
    
    # Clone the original tensor
    masked_tensor = tensor.clone()
    
    # Find indices of masked pixels
    batch_indices, channel_indices, row_indices, col_indices = torch.where(mask)
    
    # Replace masked pixels with random values from neighboring pixels
    for b, c, h, w in zip(batch_indices, channel_indices, row_indices, col_indices):
        # Create a neighborhood window with safe indexing
        h_start = max(0, h-1)
        h_end = min(H, h+2)
        w_start = max(0, w-1)
        w_end = min(W, w+2)
        
        neighborhood = tensor[b, c, h_start:h_end, w_start:w_end]
        
        # Ensure neighborhood is not empty
        if neighborhood.numel() > 0:
            # Flatten neighborhood and select random pixel
            flat_neighborhood = neighborhood.flatten()
            replacement_idx = torch.randint(len(flat_neighborhood), (1,), device=device)
            replacement = flat_neighborhood[replacement_idx]
            
            # Assign replacement value
            masked_tensor[b, c, h, w] = replacement
    
    # Return masked tensor, mask, and indices
    return masked_tensor, mask, (batch_indices, row_indices, col_indices)

def compute_n2v_loss(denoised, original, target_indices):
    """
    Compute Noise2Void loss for specific masked indices
    
    Args:
    - denoised: Denoised image tensor
    - original: Original image tensor
    - target_indices: Tuple of indices where mask was applied
    
    Returns:
    - Loss value for masked pixels
    """
    # Ensure 4D tensors
    if len(denoised.shape) == 3:
        denoised = denoised.unsqueeze(0)
        original = original.unsqueeze(0)
    
    # Unpack target indices
    batch_indices, row_indices, col_indices = target_indices
    
    # Extract masked pixel values
    # Use first channel for grayscale images
    denoised_masked = denoised[batch_indices, 0, row_indices, col_indices]
    original_masked = original[batch_indices, 0, row_indices, col_indices]
    
    # Compute MSE loss for masked pixels
    if denoised_masked.numel() > 0:
        n2v_loss = F.mse_loss(denoised_masked, original_masked)
    else:
        n2v_loss = torch.tensor(0.0, device=original.device)
    
    return n2v_loss

def create_blindspot_mask_fast(tensor, mask_ratio=0.1):
    """
    Create a blindspot mask for Noise2Void training (fast version)
    
    Args:
    - tensor: Input tensor
    - mask_ratio: Proportion of pixels to mask
    
    Returns:
    - masked_tensor: Tensor with some pixels replaced
    - mask: Boolean mask of masked pixels
    - target_indices: Indices of masked pixels
    """
    # Ensure tensor is 4D (B, C, H, W)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    B, C, H, W = tensor.shape
    device = tensor.device
    
    # Create a mask
    mask = torch.rand(B, 1, H, W, device=device) < mask_ratio
    mask = mask.expand(-1, C, -1, -1)
    
    # Clone the original tensor
    masked_tensor = tensor.clone()
    
    # Create shifted tensors in 4 directions
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    
    for dy, dx in shifts:
        # Create padded tensor
        padded = torch.nn.functional.pad(
            tensor, 
            (max(0, dx), max(0, -dx), max(0, dy), max(0, -dy)),
            mode='replicate'
        )
        
        # Get shifted values
        if dx > 0:
            padded = padded[:, :, :, :-dx]
        elif dx < 0:
            padded = padded[:, :, :, -dx:]
            
        if dy > 0:
            padded = padded[:, :, :-dy, :]
        elif dy < 0:
            padded = padded[:, :, -dy:, :]
            
        neighbors.append(padded)
    
    # Stack all neighbors
    all_neighbors = torch.stack(neighbors, dim=0)
    
    # Randomly select one neighbor for each pixel
    rand_indices = torch.randint(len(shifts), (B, 1, H, W), device=device)
    rand_indices = rand_indices.expand(-1, C, -1, -1)
    
    # Create replacement tensor
    replacements = torch.gather(
        all_neighbors, 
        0, 
        rand_indices.unsqueeze(0).expand(len(shifts), -1, -1, -1, -1)
    )[0]
    
    # Apply mask
    masked_tensor = torch.where(mask, replacements, tensor)
    
    # Find indices of masked pixels for later use
    batch_indices, channel_indices, row_indices, col_indices = torch.where(mask)
    
    # Return as in the original function but with unique pixel indices
    # (taking only indices for first channel)
    mask_points = torch.nonzero(mask[:, 0, :, :], as_tuple=True)
    batch_indices, row_indices, col_indices = mask_points
    
    return masked_tensor, mask, (batch_indices, row_indices, col_indices)