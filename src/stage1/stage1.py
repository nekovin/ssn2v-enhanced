import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\src")
from utils import normalize_image, get_stage1_loaders, get_unet_model, normalize_data

from utils import plot_loss
import torch
import time
import matplotlib.pyplot as plt
from stage1.validation import validate_n2v

import random
import numpy as np

from torch.utils.data import Dataset



def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    noise = torch.randn_like(image) * image.std() + image.mean()
    blind_input = torch.where(mask > 0, noise, blind_input)
    return blind_input

def train_stage1(img_size, model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cuda', scratch=False, mask_ratio=0.1):

    visualise = False
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if scratch:
        model = model
        history = {'train_loss': [], 'val_loss': []}
        old_epoch = 0
        print("Training from scratch")
    else:
        try:
            checkpoint = torch.load(f'checkpoints/stage1_{img_size}_best_{str(model)}_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            old_epoch = checkpoint['epoch']
            history = checkpoint['history']
            print(f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch+1}")
        except:
            print("No model found, training from scratch")
            model = model
            history = {'train_loss': [], 'val_loss': []}
            old_epoch = 0

    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Training phase
        model.train()
        running_loss = 0.0

        batch_start_time = time.time()
        
        for batch_idx, (noisy_img1, noisy_img2) in enumerate(train_loader):
            noisy_img1 = noisy_img1.to(device)
            noisy_img2 = noisy_img2.to(device)
            
            mask = torch.bernoulli(torch.full((noisy_img1.size(0), 1, noisy_img1.size(2), noisy_img1.size(3)), 
                                            mask_ratio, device=device))
            

            blind_input = create_blind_spot_input_fast(noisy_img1, mask)
            
            optimizer.zero_grad()
            
            outputs, features = model(blind_input)

            outputs = normalize_data(outputs)  
            
            loss = criterion(outputs, noisy_img1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time

            #print(f"Batch {batch_idx}/{len(train_loader)}, Time: {batch_time:.4f}s")

            batch_start_time = time.time()

        print(f"Epoch {epoch+1} finished")
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Update validation function to use N2V masking
        print("Validating")
        val_loss = validate_n2v(model, val_loader, criterion, device, mask_ratio)
        print("Validation finished")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        if visualise:
            plot_loss(history['train_loss'], history['val_loss'])
        
        if val_loss < best_val_loss:
            print(f"Saving model with val loss: {val_loss:.6f} from epoch {epoch+1}")
            best_val_loss = val_loss
            try:
                torch.save({
                    'epoch': epoch + old_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'history': history
                }, f'checkpoints/stage1_{img_size}_best_{str(model)}_model.pth')
            
            except:
                print("Err")
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {val_loss:.6f}")
        print("-" * 50)

        try:
            torch.save({
                'epoch': epoch + old_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'history': history
            }, f'checkpoints/stage1_{img_size}_last_{str(model)}_model.pth')
        except:
            print("Err")

    return model, history

def group_images_for_stage1(image_list, grouping_function=None, group_size=None):
    """
    Groups a list of already loaded images based on a grouping function or specified group size.
    
    Args:
        image_list: List of already loaded images (numpy arrays)
        grouping_function: Function that determines which group an image belongs to
                          If None, images will be grouped sequentially by group_size
        group_size: Number of images per group (used if grouping_function is None)
                   If None and grouping_function is None, defaults to 2
                   
    Returns:
        List of image groups where each group contains related noisy images
    """
    if grouping_function is None:
        # If no grouping function is provided, use sequential grouping
        if group_size is None:
            group_size = 2  # Default to pairs
        
        grouped_images = []
        for i in range(0, len(image_list), group_size):
            group = image_list[i:i+group_size]
            if len(group) >= 2:  # Only include groups with at least 2 images
                grouped_images.append(group)
    else:
        # Group images using the provided function
        groups = {}
        for i, img in enumerate(image_list):
            group_key = grouping_function(img, i)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(img)
        
        # Convert dictionary to list and filter out single-image groups
        grouped_images = [group for group in groups.values() if len(group) >= 2]
    
    return grouped_images


def process_stage1(results, epochs=10):

    flow_masks = results['flow_components']

    img_size = 256

    normalised_flow_masks = [normalize_image(flow_mask[0]) for flow_mask in flow_masks]

    flow_masks_dataset = normalised_flow_masks

    flow_masks_dataset = group_images_for_stage1(flow_masks_dataset, group_size=2)

    scratch = True

    train_loader, val_loader, test_loader = get_stage1_loaders(flow_masks_dataset, img_size)

    sample = next(iter(test_loader))

    device, model, criterion, optimizer = get_unet_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    model, history = train_stage1(img_size, model, train_loader, val_loader, criterion, optimizer, epochs, device, scratch, mask_ratio=0.50)
    model, history = train_stage1(img_size, model, train_loader, val_loader, criterion, optimizer, epochs, device, scratch, mask_ratio=0.10)
    model, history = train_stage1(img_size, model, train_loader, val_loader, criterion, optimizer, epochs, device, scratch, mask_ratio=0.25)

    return model, history