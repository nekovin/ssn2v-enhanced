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
        1.0 * foreground_loss +  # Prioritize vessel accuracy
        1.0 * background_loss +  # Strong penalty for background noise
        0.5 * edge_loss +         # Preserve vessel edges
        1.0 * conservation_loss  # Maintain physical consistency
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

def train_speckle_separation_module(input_target_data, 
                                   num_epochs=50, 
                                   batch_size=8, 
                                   learning_rate=1e-4,
                                   device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    
    # Prepare the dataset
    input_tensors = []
    target_tensors = []
    
    for input_img, target_img in input_target_data:
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
            noise_component = outputs['noise_component']

            flow_loss = loss_fn(flow_component, noise_component, batch_inputs, batch_targets)
            # Calculate loss
            # 1. Flow component should capture structures in target
            #flow_loss = nn.MSELoss()(flow_component, batch_targets)
            
            # 2. Input - noise should also resemble target
            #denoised = batch_inputs - noise_component
            denoised = batch_inputs - noise_component
            noise_loss = nn.MSELoss()(denoised, batch_targets)

            total_loss = flow_loss #+ noise_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Update running losses
            running_loss += total_loss.item()
            running_flow_loss += flow_loss.item()
            running_noise_loss += noise_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'flow_loss': flow_loss.item(),
                'noise_loss': noise_loss.item()
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
        
        # Visualize progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            # clear outputs before visualising
            from IPython.display import clear_output
            clear_output(wait=True)
            visualize_progress(model, inputs[0:1], targets[0:1], epoch+1)

        # clear visualisation
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



def apply_model_to_dataset(model_path, input_target_data, 
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare input tensors
    input_tensors = []
    
    for input_img, _ in input_target_data:  # Unpack the tuple, ignore the target
        # Convert to tensor and add channel dimension if needed
        if len(input_img.shape) == 2:
            input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
        else:
            input_tensor = torch.from_numpy(input_img).float()
            
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
            noise_component = outputs['noise_component']
            denoised = batch_inputs - noise_component
            
            # Store results (move to CPU and convert to numpy)
            results['raw_image'].append(raw_image.cpu().numpy())
            results['denoised_images'].append(denoised.cpu().numpy())
            results['flow_components'].append(flow_component.cpu().numpy())
            results['noise_components'].append(noise_component.cpu().numpy())
    
    # Concatenate batches
    for key in results:
        results[key] = np.concatenate(results[key], axis=0)
    
    return results

#results = apply_model_to_dataset("speckle_separation_model.pth", input_target_data)

def train_ssm(input_target_data):
    """
    Main function to train and test the SpeckleSeparationModule
    
    Args:
        input_target_data: List of tuples (input, target)
    """
    # Train the model
    model, history = train_speckle_separation_module(
        input_target_data,
        num_epochs=100,
        batch_size=1,
        learning_rate=1e-4
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Test the trained model
    test_trained_model(model, input_target_data, idx=0)
    
    # Save the model
    torch.save(model.state_dict(), "checkpoints/speckle_separation_model.pth")
    print("Model saved to checkpoints/speckle_separation_model.pth")

#train(input_target_data)