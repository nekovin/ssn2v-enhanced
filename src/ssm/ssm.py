import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\src")


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
    # Replace the global intensity loss with this more targeted approach
    def targeted_intensity_loss(flow_component, batch_targets):
        # Create multiple intensity level masks for different vessel brightness levels
        bright_vessel_mask = (batch_targets > 0.15).float()
        medium_vessel_mask = ((batch_targets > 0.08) & (batch_targets <= 0.15)).float()
        faint_vessel_mask = ((batch_targets > 0.03) & (batch_targets <= 0.08)).float()
        
        # Calculate separate losses for each intensity level
        bright_loss = torch.sum(torch.abs(flow_component * bright_vessel_mask - batch_targets * bright_vessel_mask)) / (bright_vessel_mask.sum() + 1e-8)
        medium_loss = torch.sum(torch.abs(flow_component * medium_vessel_mask - batch_targets * medium_vessel_mask)) / (medium_vessel_mask.sum() + 1e-8)
        faint_loss = torch.sum(torch.abs(flow_component * faint_vessel_mask - batch_targets * faint_vessel_mask)) / (faint_vessel_mask.sum() + 1e-8)
        
        # Weight them according to importance (bright vessels should match most accurately)
        return 1.5 * bright_loss + 1.0 * medium_loss + 0.7 * faint_loss
    mse = nn.MSELoss(reduction='none')

    foreground_mask = (batch_targets > 0.03).float()
    #background_mask = 1.0 - foreground_mask
    background_mask = (batch_targets < 0.1).float()
    
    pixel_wise_loss = mse(flow_component, batch_targets)
    foreground_loss = (pixel_wise_loss * foreground_mask).sum() / (foreground_mask.sum() + 1e-8)
    
    #background_loss = torch.mean((flow_component * background_mask)**2)
    background_loss = torch.mean((flow_component * background_mask)**2) * 2.5

    horizontal_kernel = torch.tensor([[[[1, -1]]]], dtype=batch_targets.dtype, device=batch_targets.device)
    vertical_kernel = torch.tensor([[[[1], [-1]]]], dtype=batch_targets.dtype, device=batch_targets.device)
    diagonal1_kernel = torch.tensor([[[[1, 0], [0, -1]]]], dtype=batch_targets.dtype, device=batch_targets.device)
    diagonal2_kernel = torch.tensor([[[[0, 1], [-1, 0]]]], dtype=batch_targets.dtype, device=batch_targets.device)

    # Calculate edges in horizontal direction
    h_edges_target = torch.abs(torch.nn.functional.conv2d(batch_targets, horizontal_kernel, padding='same'))
    h_edges_flow = torch.abs(torch.nn.functional.conv2d(flow_component, horizontal_kernel, padding='same'))
    h_edge_loss = mse(h_edges_flow, h_edges_target).mean()

    # Calculate edges in vertical direction
    v_edges_target = torch.abs(torch.nn.functional.conv2d(batch_targets, vertical_kernel, padding='same'))
    v_edges_flow = torch.abs(torch.nn.functional.conv2d(flow_component, vertical_kernel, padding='same'))
    v_edge_loss = mse(v_edges_flow, v_edges_target).mean()

    # Calculate edges in diagonal directions
    d1_edges_target = torch.abs(torch.nn.functional.conv2d(batch_targets, diagonal1_kernel, padding='same'))
    d1_edges_flow = torch.abs(torch.nn.functional.conv2d(flow_component, diagonal1_kernel, padding='same'))
    d1_edge_loss = mse(d1_edges_flow, d1_edges_target).mean()

    d2_edges_target = torch.abs(torch.nn.functional.conv2d(batch_targets, diagonal2_kernel, padding='same'))
    d2_edges_flow = torch.abs(torch.nn.functional.conv2d(flow_component, diagonal2_kernel, padding='same'))
    d2_edge_loss = mse(d2_edges_flow, d2_edges_target).mean()

    # Combine all edge losses
    edge_loss = (h_edge_loss + v_edge_loss + d1_edge_loss + d2_edge_loss) / 4.0

    targeted_intensity_loss = targeted_intensity_loss(flow_component, batch_targets)
    
    #conservation_loss = nn.MSELoss()(batch_inputs, flow_component + noise_component)
    conservation_loss = nn.MSELoss()(batch_targets, flow_component)
    
    
    total_loss = (
        1.0 * foreground_loss +  # Prioritize vessel accuracy
        1.0 * background_loss +  # Strong penalty for background noise
        1.0 * edge_loss +         # Preserve vessel edges
        1.0 * conservation_loss  # Maintain physical consistency
        + 1.0 * targeted_intensity_loss  # Encourage intensity matching
    )
    
    return total_loss

def custom_loss(flow_component, noise_component, batch_inputs, batch_targets):
    mse = nn.MSELoss(reduction='none')

    # Create masks with slightly more aggressive thresholding
    foreground_mask = (batch_targets > 0.04).float()
    background_mask = 1.0 - foreground_mask
    
    # Stronger focus on vessel regions
    pixel_wise_loss = mse(flow_component, batch_targets)
    foreground_loss = (pixel_wise_loss * foreground_mask).sum() / (foreground_mask.sum() + 1e-8)
    
    # More aggressive background suppression
    background_loss = torch.mean((flow_component * background_mask)**2) * 2.0
    
    # Edge preservation (keep your existing code)
    horizontal_kernel = torch.tensor([[[[1, -1]]]], dtype=batch_targets.dtype, device=batch_targets.device)
    vertical_kernel = torch.tensor([[[[1], [-1]]]], dtype=batch_targets.dtype, device=batch_targets.device)
    diagonal1_kernel = torch.tensor([[[[1, 0], [0, -1]]]], dtype=batch_targets.dtype, device=batch_targets.device)
    diagonal2_kernel = torch.tensor([[[[0, 1], [-1, 0]]]], dtype=batch_targets.dtype, device=batch_targets.device)

    # Edge losses calculation (keep your existing code)
    h_edges_target = torch.abs(torch.nn.functional.conv2d(batch_targets, horizontal_kernel, padding='same'))
    h_edges_flow = torch.abs(torch.nn.functional.conv2d(flow_component, horizontal_kernel, padding='same'))
    h_edge_loss = mse(h_edges_flow, h_edges_target).mean()
    
    # Same for v_edge_loss, d1_edge_loss, d2_edge_loss (keep your existing code)
    v_edges_target = torch.abs(torch.nn.functional.conv2d(batch_targets, vertical_kernel, padding='same'))
    v_edges_flow = torch.abs(torch.nn.functional.conv2d(flow_component, vertical_kernel, padding='same'))
    v_edge_loss = mse(v_edges_flow, v_edges_target).mean()

    d1_edges_target = torch.abs(torch.nn.functional.conv2d(batch_targets, diagonal1_kernel, padding='same'))
    d1_edges_flow = torch.abs(torch.nn.functional.conv2d(flow_component, diagonal1_kernel, padding='same'))
    d1_edge_loss = mse(d1_edges_flow, d1_edges_target).mean()

    d2_edges_target = torch.abs(torch.nn.functional.conv2d(batch_targets, diagonal2_kernel, padding='same'))
    d2_edges_flow = torch.abs(torch.nn.functional.conv2d(flow_component, diagonal2_kernel, padding='same'))
    d2_edge_loss = mse(d2_edges_flow, d2_edges_target).mean()

    # Combined edge loss
    edge_loss = (h_edge_loss + v_edge_loss + d1_edge_loss + d2_edge_loss) / 4.0
    
    # Add intensity differentiation loss to prevent uniform intensity changes
    # This encourages relative intensity differences between vessel pixels
    intensity_variation_loss = -torch.var(flow_component * foreground_mask) * 0.5
    
    # Add stronger focus on matching high-intensity vessel regions
    bright_vessel_mask = (batch_targets > 0.1).float()
    bright_vessel_loss = (mse(flow_component, batch_targets) * bright_vessel_mask).sum() / (bright_vessel_mask.sum() + 1e-8)
    
    # Final loss with reweighted components
    total_loss = (
        1.2 * foreground_loss +      # Increased focus on vessel regions
        1.5 * background_loss +      # Stronger background suppression
        1.0 * edge_loss +            # Maintain edge preservation
        0.5 * intensity_variation_loss + # Encourage intensity variations
        0.8 * bright_vessel_loss     # Extra emphasis on brightest vessels
    )
    
    return total_loss

class SpeckleSeparationModule(nn.Module):
    """
    A simplified module to separate OCT speckle into:
    - Informative speckle (related to blood flow)
    - Noise speckle (to be removed)
    """
    
    def __init__(self, input_channels=1, feature_dim=32):
        """
        Initialize the Speckle Separation Module
        
        Args:
            input_channels: Number of input image channels (default: 1 for grayscale OCT)
            feature_dim: Dimension of feature maps
        """
        super(SpeckleSeparationModule, self).__init__()
        
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Flow component branch
        self.flow_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1),
            nn.Tanh()
        )
        
        # Noise component branch
        self.noise_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass of the Speckle Separation Module
        
        Args:
            x: Input OCT image tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary containing:
                - 'flow_component': Flow-related speckle component
                - 'noise_component': Noise-related speckle component
        """
        # Extract features
        features = self.feature_extraction(x)
        
        # Separate into flow and noise components
        flow_component = self.flow_branch(features)
        noise_component = self.noise_branch(features)
        
        return {
            'flow_component': flow_component,
            'noise_component': noise_component
        }
    
######################
    
class SpeckleSeparationUNet(nn.Module):
    """
    Enhanced deeper U-Net architecture for OCT speckle separation
    """
    
    def __init__(self, input_channels=1, feature_dim=32, depth=5, block_depth=3):
        """
        Initialize the Deeper Speckle Separation U-Net Module
        
        Args:
            input_channels: Number of input image channels (default: 1 for grayscale OCT)
            feature_dim: Initial dimension of feature maps
            depth: Depth of the U-Net (number of downsampling/upsampling operations)
            block_depth: Number of convolution layers in each encoder/decoder block
        """
        super(SpeckleSeparationUNet, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.depth = depth
        
        # Encoder path with deeper blocks
        in_channels = input_channels
        for i in range(depth):
            out_channels = feature_dim * (2**min(i, 3))  # Cap feature growth to avoid excessive memory usage
            encoder_block = []
            
            # First conv in the block
            encoder_block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            encoder_block.append(nn.BatchNorm2d(out_channels))
            encoder_block.append(nn.ReLU(inplace=True))
            
            # Additional conv layers in each block
            for _ in range(block_depth - 1):
                encoder_block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                encoder_block.append(nn.BatchNorm2d(out_channels))
                encoder_block.append(nn.ReLU(inplace=True))
            
            self.encoder_blocks.append(nn.Sequential(*encoder_block))
            in_channels = out_channels
        
        # Bottleneck
        bottleneck_channels = feature_dim * (2**min(depth, 3))
        bottleneck = []
        for _ in range(block_depth + 1):  # Slightly deeper bottleneck
            bottleneck.append(nn.Conv2d(in_channels, bottleneck_channels, kernel_size=3, padding=1))
            bottleneck.append(nn.BatchNorm2d(bottleneck_channels))
            bottleneck.append(nn.ReLU(inplace=True))
            in_channels = bottleneck_channels
        
        self.bottleneck = nn.Sequential(*bottleneck)
        
        # Decoder path with deeper blocks
        in_channels = bottleneck_channels
        for i in range(depth):
            out_channels = feature_dim * (2**min(depth-i-1, 3))
            decoder_block = []
            
            # First conv after the skip connection
            decoder_block.append(nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1))
            decoder_block.append(nn.BatchNorm2d(out_channels))
            decoder_block.append(nn.ReLU(inplace=True))
            
            # Additional conv layers in each block
            for _ in range(block_depth - 1):
                decoder_block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                decoder_block.append(nn.BatchNorm2d(out_channels))
                decoder_block.append(nn.ReLU(inplace=True))
            
            self.decoder_blocks.append(nn.Sequential(*decoder_block))
            in_channels = out_channels
        
        # Output layers with residual connections
        self.flow_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1)
        )
        
        self.noise_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1)
        )
        
        # Upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Add dilated convolutions for wider receptive field
        self.dilation_block = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass of the Speckle Separation U-Net
        
        Args:
            x: Input OCT image tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary containing:
                - 'flow_component': Flow-related speckle component
                - 'noise_component': Noise-related speckle component
        """
        # Store encoder outputs for skip connections
        encoder_features = []
        
        # Encoder path
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            encoder_features.append(x)
            if i < self.depth - 1:
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i in range(self.depth):
            x = self.up(x)
            # Ensure matching sizes for concatenation
            encoder_feature = encoder_features[self.depth - i - 1]
            if x.size() != encoder_feature.size():
                x = nn.functional.interpolate(x, size=encoder_feature.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decoder_blocks[i](x)
        
        # Apply dilated convolutions for larger receptive field
        #x = self.dilation_block(x)
        
        # Generate flow and noise components
        flow_component = self.flow_branch(x)
        noise_component = self.noise_branch(x)
        
        return {
            'flow_component': flow_component,
            'noise_component': noise_component
        }

########################

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

    def adaptive_threshold(denoised_img, sensitivity=0.02):
        # Get an estimate of noise level from background
        bg_mask = denoised_img < np.percentile(denoised_img, 50)
        noise_std = np.std(denoised_img[bg_mask])
        
        # Set threshold as a multiple of background noise
        threshold = sensitivity * noise_std
        return np.maximum(denoised_img - threshold, 0)
    
    # Get components
    input_np = input_tensor[0, 0].cpu().numpy()
    target_np = target_tensor[0, 0].cpu().numpy()
    flow_np = output['flow_component'][0, 0].cpu().numpy()
    noise_np = output['noise_component'][0, 0].cpu().numpy()
    denoised_np = input_np - noise_np
    denoised_np = adaptive_threshold(denoised_np, sensitivity=0.02)
    
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
    #
    axes[1, 0].imshow(flow_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("Flow Component")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(noise_np, cmap='gray')
    axes[1, 1].set_title("Noise Component")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(denoised_np, cmap='gray')
    axes[1, 2].set_title("Denoised (Input - Noise)")
    axes[1, 2].axis('off')

    flow_np = normalize_image(flow_np)
    axes[0, 2].imshow(flow_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title("Flow Component (Normalized)")
    axes[0, 2].axis('off')

    
    plt.suptitle(f"Training Progress - Epoch {epoch}")
    plt.tight_layout()
    plt.show()

from IPython.display import clear_output

def train_speckle_separation_module(dataset, 
                                   num_epochs=50, 
                                   batch_size=8, 
                                   learning_rate=1e-4,
                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                   unet=False):
    """
    Train the SpeckleSeparationModule using the provided input-target data
    
    Args:
        input_target_data: List of tuples (input, target) where both are numpy arrays
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    print(f"Using device: {device}")


    torch.autograd.set_detect_anomaly(True)
    
    # Prepare the dataset
    input_tensors = []
    target_tensors = []
    
    for patient in dataset:
        patient_data = dataset[patient]
        for input_img, target_img in patient_data:
            # Convert to tensor and add channel dimension if needed
            if len(input_img.shape) == 2:
                input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
                target_tensor = torch.from_numpy(target_img).float().unsqueeze(0)
            else:
                input_tensor = torch.from_numpy(input_img).float()
                target_tensor = torch.from_numpy(target_img).float()
                
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
        
    # Stack into batch dimension
    inputs = torch.stack(input_tensors).to(device)
    targets = torch.stack(target_tensors).to(device)

    #loss_fn = Noise2VoidLoss(alpha=0.8, beta=0.1, gamma=0.1)
    loss_fn = custom_loss
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create the model
    if unet:
        model = SpeckleSeparationUNet(input_channels=1, feature_dim=32).to(device)
    else:
        model = SpeckleSeparationModule(input_channels=1, feature_dim=32).to(device)

    #model.load_state_dict(torch.load(r"checkpoints/speckle_separation_model.pth", map_location=device))

    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'loss': [],
        'flow_loss': [],
        'noise_loss': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_flow_loss = 0.0
        running_noise_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_inputs, batch_targets in progress_bar:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_inputs)


            flow_component = outputs['flow_component']
            #flow_component = normalize_image(flow_component)
            noise_component = outputs['noise_component']
            #noise_component = normalize_image(noise_component)

            total_loss = loss_fn(flow_component, noise_component, batch_inputs, batch_targets)

            print(total_loss)
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            noise_loss = 0
            
            # Update running losses
            running_loss += total_loss
            running_flow_loss += total_loss.item()
            running_noise_loss += 0 #noise_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'flow_loss': total_loss.item(),
                'noise_loss': noise_loss
            })
        
        # Calculate average losses for the epoch
        avg_loss = running_loss / len(dataloader)
        avg_flow_loss = running_flow_loss / len(dataloader)
        avg_noise_loss = running_noise_loss / len(dataloader)
        
        # Update history
        history['loss'].append(avg_loss)
        history['flow_loss'].append(avg_flow_loss)
        history['noise_loss'].append(avg_noise_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Flow Loss: {avg_flow_loss:.6f}, Noise Loss: {avg_noise_loss:.6f}")
        
        clear_output(wait=True)
        visualize_progress(model, inputs[0:1], targets[0:1], epoch+1)

        plt.close()
    
    return model, history

def train_speckle_separation_module_n2n(dataset, 
                                   num_epochs=50, 
                                   batch_size=8, 
                                   learning_rate=1e-4,
                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                   unet=False):
    """
    Train the SpeckleSeparationModule using the provided input-target data
    
    Args:
        input_target_data: List of tuples (input, target) where both are numpy arrays
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    print(f"Using device: {device}")


    torch.autograd.set_detect_anomaly(True)
    
    # Prepare the dataset
    input_tensors = []
    target_tensors = []
    
    for patient in dataset:
        patient_data = dataset[patient]
        for input_img, target_img in patient_data:
            # Convert to tensor and add channel dimension if needed
            if len(input_img.shape) == 2:
                input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
                target_tensor = torch.from_numpy(target_img).float().unsqueeze(0)
            else:
                input_tensor = torch.from_numpy(input_img).float()
                target_tensor = torch.from_numpy(target_img).float()
                
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
        
    # Stack into batch dimension
    inputs = torch.stack(input_tensors).to(device)
    targets = torch.stack(target_tensors).to(device)

    # Replace with n2v_loss function
    loss_fn = custom_loss
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create the model
    if unet:
        model = SpeckleSeparationUNet(input_channels=1, feature_dim=32).to(device)
    else:
        model = SpeckleSeparationModule(input_channels=1, feature_dim=32).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'loss': [],
        'flow_loss': [],
        'noise_loss': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_flow_loss = 0.0
        running_noise_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        print("Training...")
        for batch_inputs, batch_targets in progress_bar:
            # Zero the gradients
            optimizer.zero_grad()
            
            # CREATE MASKED INPUT - NEW CODE FOR N2V
            # Randomly select pixels to mask
            mask = torch.rand_like(batch_inputs) > 0.75  # Mask ~25% of pixels
            
            # Make a copy of inputs to modify
            masked_inputs = batch_inputs.clone()
            
            # Replace masked pixels with random nearby pixel values
            roll_amount = torch.randint(-5, 5, (2,))
            shifted = torch.roll(batch_inputs, shifts=(roll_amount[0].item(), roll_amount[1].item()), dims=(2, 3))
            masked_inputs[mask] = shifted[mask]
            
            # Forward pass WITH MASKED INPUTS
            outputs = model(masked_inputs)
            # END OF NEW CODE

            flow_component = outputs['flow_component']
            noise_component = outputs['noise_component']

            # MODIFIED LOSS CALCULATION FOR N2V
            # N2V loss - prediction should match original values at masked positions
            mse = nn.MSELoss(reduction='none')
            n2v_loss = mse(flow_component[mask], batch_targets[mask]).mean()
            
            # Regular loss components (from your custom_loss)
            total_loss = loss_fn(flow_component, noise_component, batch_inputs, batch_targets)
            
            # Add N2V loss component
            total_loss = total_loss + 1.2 * n2v_loss
            # END OF MODIFIED LOSS CALCULATION

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            noise_loss = 0
            
            # Update running losses
            running_loss += total_loss
            running_flow_loss += total_loss.item()
            running_noise_loss += 0 #noise_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'flow_loss': total_loss.item(),
                'noise_loss': noise_loss
            })
        
        # Calculate average losses for the epoch
        avg_loss = running_loss / len(dataloader)
        avg_flow_loss = running_flow_loss / len(dataloader)
        avg_noise_loss = running_noise_loss / len(dataloader)
        
        # Update history
        history['loss'].append(avg_loss)
        history['flow_loss'].append(avg_flow_loss)
        history['noise_loss'].append(avg_noise_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Flow Loss: {avg_flow_loss:.6f}, Noise Loss: {avg_noise_loss:.6f}")
        
        clear_output(wait=True)
        visualize_progress(model, inputs[0:1], targets[0:1], epoch+1)

        plt.close()
    
    return model, history

def test_trained_model(model, input_target_data, idx=0, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Test a trained model on a specific image pair
    
    Args:
        model: Trained SpeckleSeparationModule
        input_target_data: List of tuples (input, target)
        idx: Index of the data pair to use
        device: Device to run inference on
    """
    # Get the test image pair
    input_img = input_target_data[idx][0]
    target_img = input_target_data[idx][1]
    
    # Convert to tensor
    if len(input_img.shape) == 2:
        input_tensor = torch.from_numpy(input_img).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        input_tensor = torch.from_numpy(input_img).float().unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    flow = output['flow_component'][0, 0].cpu().numpy()
    noise = output['noise_component'][0, 0].cpu().numpy()
    denoised = input_img - noise
    
    #visualize_progress(model, input_img, target_img, 0)
    
    # PSNR between denoised and target
    try:
        psnr = peak_signal_noise_ratio(target_img, denoised)
        print(f"PSNR: {psnr:.2f} dB")
    except:
        print("Could not calculate PSNR")
    
    # SSIM between denoised and target
    try:
        ssim = structural_similarity(target_img, denoised)
        print(f"SSIM: {ssim:.4f}")
    except:
        print("Could not calculate SSIM")
    
    return {
        'flow': flow,
        'noise': noise,
        'denoised': denoised
    }

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
    thresholded = (octa > threshold).astype(float)
    
    # Convert back to tensor if needed
    if not torch.is_tensor(octa):
        thresholded = torch.from_numpy(thresholded).float()
    
    return thresholded

def apply_model_to_dataset(model_path, dataset, 
                          batch_size=8, 
                          device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load a trained SpeckleSeparationModule and apply it to the input data
    
    Args:
        model_path: Path to the saved model
        input_target_data: List of tuples (input, target) where both are numpy arrays
        batch_size: Batch size for processing
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing denoised images, flow components, and noise components
    """
    print(f"Using device: {device}")
    
    # Load the model
    model = SpeckleSeparationModule(input_channels=1, feature_dim=32)
    #model = SpeckleSeparationUNet(input_channels=1, feature_dim=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare input tensors
    input_tensors = []
    
    dataset_list = []
    for i in dataset.keys():
        dataset_list.append(dataset[i])
    
    '''
    for input_img, _ in input_target_data:  # Unpack the tuple, ignore the target
        # Convert to tensor and add channel dimension if needed
        if len(input_img.shape) == 2:
            input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
        else:
            input_tensor = torch.from_numpy(input_img).float()
            
        input_tensors.append(input_tensor)'''

    flattened_dataset = []
    for patient_data in dataset_list:
        if isinstance(patient_data, list):
            flattened_dataset.extend(patient_data)
        else:
            flattened_dataset.append(patient_data)

    input_tensors = []
    for input_img in flattened_dataset:
        # Convert to tensor and add channel dimension if needed
        if isinstance(input_img, list):
            input_img = np.array(input_img)
                
        if len(input_img.shape) == 2:
            # For 2D images, add a single channel
            input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)  # [H, W] -> [1, H, W]
        elif len(input_img.shape) == 3:
            # For 3D images, ensure we're using only the first channel if there are multiple
            if input_img.shape[0] > 1 or input_img.shape[2] > 1:  # Checking if first or last dim is channel
                # Take only the first channel
                if input_img.shape[0] > 1:  # If channels are in first dimension
                    input_img = input_img[0:1, :, :]
                elif input_img.shape[2] > 1:  # If channels are in last dimension
                    input_img = input_img[:, :, 0:1]
                    input_img = np.transpose(input_img, (2, 0, 1))  # [H, W, 1] -> [1, H, W]
            input_tensor = torch.from_numpy(input_img).float()
        else:
            # Handle unexpected dimensions
            raise ValueError(f"Unexpected image shape: {input_img.shape}")
                
        input_tensors.append(input_tensor)
        
    # Stack into batch dimension
    inputs = torch.stack(input_tensors).to(device)
    
    # Create dataloader
    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Results containers
    results = {
        'raw_image': [],
        'denoised_images': [],
        'flow_components': [],
        'noise_components': []
    }
    
    # Process the dataset
    with torch.no_grad():
        for (batch_inputs,) in tqdm(dataloader, desc="Processing dataset"):
            # Apply model
            outputs = model(batch_inputs)
            
            raw_image = batch_inputs
            flow_component = outputs['flow_component']
            flow_component = normalize_image(flow_component.cpu().numpy())
            noise_component = outputs['noise_component']
            denoised = batch_inputs - noise_component
            
            # Store results (move to CPU and convert to numpy)
            results['raw_image'].append(raw_image.cpu().numpy())
            results['denoised_images'].append(denoised.cpu().numpy())
            #results['flow_components'].append(flow_component)
            #flow_component = threshold_octa(flow_component, method='percentile', threshold_percentage=0.01)
            results['flow_components'].append(flow_component)
            results['noise_components'].append(noise_component.cpu().numpy())
    
    # Concatenate batches
    for key in results:
        results[key] = np.concatenate(results[key], axis=0)

    print("Length of results:")
    for key in results:
        print(f"{key}: {len(results[key])}")
    
    return results

#results = apply_model_to_dataset("speckle_separation_model.pth", input_target_data)

def train_ssm(dataset, unet):
    """
    Main function to train and test the SpeckleSeparationModule
    
    Args:
        input_target_data: List of tuples (input, target)
    """
    # Train the model
    model, history = train_speckle_separation_module(
        dataset,
        num_epochs=200,
        batch_size=1,
        learning_rate=1e-4,
        unet=False
    )
    
    # Plot training history
    #plot_training_history(history)
    
    # Test the trained model
    #test_trained_model(model, dataset, idx=0)
    
    # Save the model
    torch.save(model.state_dict(), "checkpoints/speckle_separation_model_unet.pth")
    print("Model saved to checkpoints/speckle_separation_model_unet.pth")

#train(input_target_data)