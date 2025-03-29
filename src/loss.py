import torch
import torch.nn as nn

class Noise2VoidLoss(nn.Module):
    def __init__(self, mask_ratio=0.1):
        super(Noise2VoidLoss, self).__init__()
        self.mask_ratio = mask_ratio
        self.mse = nn.MSELoss(reduction='none')  # Changed to 'none' to handle per-pixel loss

    def forward(self, prediction, target):
        mask = torch.bernoulli(torch.full_like(target, self.mask_ratio)) # static mask

        pixel_losses = self.mse(prediction, target)
        
        masked_losses = pixel_losses * mask
        loss = masked_losses.sum() / (mask.sum() + 1e-6) # 1e-8 is too big

        #visualize_masks(self, mask, prediction, target)
            
        return loss