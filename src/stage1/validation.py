import torch
import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\src")
from utils import normalize_data
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from tqdm import tqdm

def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    noise = torch.randn_like(image) * image.std() + image.mean()
    blind_input = torch.where(mask > 0, noise, blind_input)
    return blind_input

def visualise_n2v(blind_input, target_img, output, mask=None):
    """
    Visualize the N2V process with mask overlay
    
    Args:
        blind_input: Input with blind spots
        target_img: Target noisy image
        output: Model prediction
        mask: Binary mask showing pixel positions for N2V
    """
    # Normalize output to match input scale
    output = normalize_data(output)
    
    clear_output(wait=True)
    
    # If mask is provided, show 4 images including mask
    if mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # Plot blind spot input
        axes[0].imshow(blind_input.squeeze(), cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Blind-spot Input')
        
        # Plot model output
        axes[1].imshow(output.squeeze(), cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Output Image')
        
        # Plot target image
        axes[2].imshow(target_img.squeeze(), cmap='gray')
        axes[2].axis('off')
        axes[2].set_title('Target Noisy Image')
        
        # Plot mask overlay (red points on original image)
        axes[3].imshow(target_img.squeeze(), cmap='gray')
        
        # Create mask overlay as red dots
        mask_overlay = np.zeros((*mask.squeeze().shape, 4))  # RGBA
        mask_overlay[mask.squeeze() > 0] = [1, 0, 0, 0.7]  # Red with alpha
        axes[3].imshow(mask_overlay)
        axes[3].axis('off')
        axes[3].set_title('Mask Overlay (red)')
    
    # Otherwise use original 3-image layout
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].imshow(blind_input.squeeze(), cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Blind-spot Input')
        
        axes[1].imshow(output.squeeze(), cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Output Image')
        
        axes[2].imshow(target_img.squeeze(), cmap='gray')
        axes[2].axis('off')
        axes[2].set_title('Target Noisy Image')
    
    plt.tight_layout()
    plt.show()

def validate_n2v(model, val_loader, criterion, device='cuda', mask_ratio=0.1, visualise=False):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for noisy_img1, noisy_img2 in val_loader:
            noisy_img1 = noisy_img1.to(device)
            noisy_img2 = noisy_img2.to(device)

            mask = torch.bernoulli(torch.full((noisy_img1.size(0), 1, noisy_img1.size(2), noisy_img1.size(3)), 
                                          mask_ratio, device=device))
            
            # Create blind spot input
            blind_input = create_blind_spot_input_fast(noisy_img1, mask)
            
            try:
                outputs, features = model(blind_input)
            except:
                outputs = model(blind_input)
                
            outputs = normalize_data(outputs)  
            loss = criterion(outputs, noisy_img1)
            
            total_loss += loss.item()
            
            if visualise:
                visualise_n2v(
                    blind_input.cpu().detach().numpy(),
                    noisy_img1.cpu().detach().numpy(),
                    outputs.cpu().detach().numpy(),
                    mask.cpu().detach().numpy()  # Pass the mask to visualization
                )
    
    return total_loss / len(val_loader)