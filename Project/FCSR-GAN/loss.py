
import torch
import torch.nn as nn
import torch.nn.functional as F

def occlusion_aware_l1(fake, real, mask):
    # Upsample fake to match real size if needed
    if fake.shape[2:] != real.shape[2:]:
        fake = F.interpolate(fake, size=real.shape[2:], mode='bilinear', align_corners=False)

    # Resize mask to match real
    mask_resized = F.interpolate(mask, size=real.shape[2:], mode='nearest')
    mask_resized = mask_resized.repeat(1, real.shape[1], 1, 1)  # repeat for channels

    # Compute weighted L1 loss
    weights = torch.ones_like(real)
    weights[mask_resized == 1] = 5.0
    weighted_diff = torch.abs(fake - real) * weights
    return weighted_diff.mean()
