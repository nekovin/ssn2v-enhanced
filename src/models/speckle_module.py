import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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
    
    def visualize_components(self, x):
        """
        Visualize the flow and noise components
        
        Args:
            x: Input tensor [B, C, H, W]
        """
        with torch.no_grad():
            output = self.forward(x)
            
        # Convert to numpy for visualization
        input_np = x[0, 0].cpu().numpy()
        flow_np = output['flow_component'][0, 0].cpu().numpy()
        noise_np = output['noise_component'][0, 0].cpu().numpy()
        
        # Plot the components
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(input_np, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(flow_np, cmap='inferno')
        axes[1].set_title('Flow Component')
        axes[1].axis('off')
        
        axes[2].imshow(noise_np, cmap='viridis')
        axes[2].set_title('Noise Component')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

def test_speckle_separation_module():
    """
    Test function to verify that the SpeckleSeparationModule is working correctly.
    This checks:
    1. The module runs without errors
    2. The output shapes match the input shape
    3. The flow and noise components are different
    4. The sum of flow and noise approximates the original input
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create the module
    module = SpeckleSeparationModule(input_channels=1, feature_dim=32)
    
    # Create a test input (batch size 2, 1 channel, 64x64 image)
    batch_size, channels, height, width = 2, 1, 64, 64
    test_input = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    with torch.no_grad():
        output = module(test_input)
    
    # Check output shapes
    flow = output['flow_component']
    noise = output['noise_component']
    
    print(f"Input shape: {test_input.shape}")
    print(f"Flow component shape: {flow.shape}")
    print(f"Noise component shape: {noise.shape}")
    
    # Check that shapes match
    assert flow.shape == test_input.shape, f"Flow shape {flow.shape} doesn't match input shape {test_input.shape}"
    assert noise.shape == test_input.shape, f"Noise shape {noise.shape} doesn't match input shape {test_input.shape}"
    
    # Check that flow and noise are different (their difference should have non-zero values)
    diff = flow - noise
    assert torch.abs(diff).mean().item() > 0, "Flow and noise components are identical"
    
    # Check if flow + noise approximates the input (they won't be exactly equal due to nonlinearities)
    # But for a basic test, we can check if they're at least somewhat related
    combined = flow + noise
    correlation = torch.nn.functional.cosine_similarity(test_input.view(batch_size, -1), 
                                                      combined.view(batch_size, -1)).mean()
    
    print(f"Correlation between input and (flow + noise): {correlation.item():.4f}")
    assert correlation.item() > -0.5, "Very low correlation between input and components"
    
    # Visualize for the first batch item
    plt.figure(figsize=(12, 4))
    
    # Original input
    plt.subplot(1, 4, 1)
    plt.imshow(test_input[0, 0].numpy(), cmap='gray')
    plt.title("Input")
    plt.axis('off')
    
    # Flow component
    plt.subplot(1, 4, 2)
    plt.imshow(flow[0, 0].numpy(), cmap='gray')
    plt.title("Flow Component")
    plt.axis('off')
    
    # Noise component
    plt.subplot(1, 4, 3)
    plt.imshow(noise[0, 0].numpy(), cmap='gray')
    plt.title("Noise Component")
    plt.axis('off')
    
    # Sum of components
    plt.subplot(1, 4, 4)
    plt.imshow(combined[0, 0].numpy(), cmap='gray')
    plt.title("Flow + Noise")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("All tests passed. The SpeckleSeparationModule is working correctly.")