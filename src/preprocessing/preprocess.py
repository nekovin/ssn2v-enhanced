import os
import glob
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from skimage import io
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

def load_patient_data(base_path):
    """
    Load OCT B-scans for a patient from the specified path.
    
    Args:
        base_path: Path to the directory containing the OCT images
        
    Returns:
        List of loaded OCT scans as normalized numpy arrays
    """
    print(f"Loading data from: {base_path}")
    
    # Find all TIFF files in the directory
    files = sorted(glob.glob(os.path.join(base_path, "*.tiff")))
    if not files:
        # Try other possible extensions if no .tiff files are found
        files = sorted(glob.glob(os.path.join(base_path, "*.tif")))
    if not files:
        files = sorted(glob.glob(os.path.join(base_path, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(base_path, "*.jpg")))
        
    print(f"Found {len(files)} files")
    
    # Load and preprocess images
    oct_scans = []
    for file in files:
        try:
            # Read the image
            img = io.imread(file)
            
            # Convert to float32 and normalize to [0, 1]
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
                
            oct_scans.append(img)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return oct_scans


def standard_preprocessing(oct_volume):
    preprocessed = []
    for img in oct_volume:
        # Normalize to [0, 1] if not already, just incase
        if img.max() > 1.0:
            img = img / 255.0
        
        # Add channel dimension if needed
        if len(img.shape) == 2:
            img_with_channel = img.reshape(*img.shape, 1)
        else:
            img_with_channel = img
        
        resized = cv2.resize(img_with_channel, (256, 256), interpolation=cv2.INTER_LINEAR)
        preprocessed.append(resized)

    return np.array(preprocessed)

def octa_preprocessing(preprocessed_data):
    n_scans = len(preprocessed_data)
    pairs = []
    for i in range(n_scans - 1):
        pairs.append(preprocessed_data[i])
        pairs.append(preprocessed_data[i+1])
    
    # Compute OCTA images from OCT pairs
    octa_images = []
    for i in range(0, len(pairs), 2):
        if i+1 < len(pairs):
            octa = compute_octa(pairs[i], pairs[i+1])
            
            thresholded_octa = threshold_octa(octa, pairs[i])
            
            octa_images.append(thresholded_octa)
    
    #octa_batch = np.array(octa_images)

    return octa_images #list

def compute_octa(oct1, oct2):

    numerator = (oct1 - oct2)**2
    denominator = oct1**2 + oct2**2
    
    epsilon = 1e-10
    octa = numerator / (denominator + epsilon)

    #octa = (octa - tf.reduce_min(octa)) / (tf.reduce_max(octa) - tf.reduce_min(octa) + epsilon)
    #octa = octa * 0.014  # Match the max value of target OCTA
    #octa = np.array(octa)
    
    return octa

def threshold_octa(octa, oct):

    background_mask = oct < np.percentile(oct, 95)  # Bottom 20% of OCT values
    
    if np.sum(background_mask) > 0:  # Ensure we have background pixels
        background_mean = np.mean(oct[background_mask])
        background_std = np.std(oct[background_mask])
        
        threshold = background_mean + 2 * background_std
    else:

        threshold = np.percentile(oct, 1)
    
    signal_mask = oct > threshold
    
    thresholded_octa = octa * signal_mask
    
    return thresholded_octa


def pair_data(preprocessed_data, octa_data):
    preprocessed_data = preprocessed_data[:-1]  # Remove last scan to avoid out of bounds

    input_target = []
    for p, o in zip(preprocessed_data, octa_data):
        input_target.append([p, o])

    return input_target

def preprocessing():
    try:    
        i = 1
        data = load_patient_data(rf"C:\Datasets\ICIP training data\ICIP training data\0\RawDataQA ({i})")

        preprocessed_data = standard_preprocessing(data)

        octa_data = octa_preprocessing(preprocessed_data)

        input_target_data = pair_data(preprocessed_data, octa_data)

        return input_target_data
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")

    

