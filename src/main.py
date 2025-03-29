import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\src")
from preprocessing.preprocess import preprocessing
from ssm.ssm import train_ssm, apply_model_to_dataset
from stage1.stage1 import process_stage1
from stage2.stage2 import process_stage2
from PIL import Image
import numpy as np

def main():
    input_target_data = preprocessing()
    #train_ssm(input_target_data) # comment out if unused
    results = apply_model_to_dataset("checkpoints/speckle_separation_model.pth", input_target_data)
    #model, history = process_stage1(results, epochs=10)
    denoised_oct1, denoised_oct2 = process_stage2(results, epochs=3)

    # Save the denoised images
    os.makedirs('results', exist_ok=True)
    #denoised_oct1.save(os.path.join('results', 'denoised_oct1.png'))
    Image.fromarray((denoised_oct1.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join('results', 'denoised_oct1.png'))
    Image.fromarray((denoised_oct2.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join('results', 'denoised_oct2.png'))

if __name__ == "__main__":
    main()