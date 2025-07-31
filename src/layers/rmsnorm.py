"""
RMSNorm (Root Mean Square Layer Normalization)
More stable and efficient normalization layer, standard normalization method used by DeepSeek
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm implementation for numerical stability.
    
    RMSNorm normalizes using only the root mean square, without mean subtraction.
    This is more numerically stable and efficient than LayerNorm.
    
    Args:
        d (int): Hidden dimension
        eps (float): Small epsilon for numerical stability
    """
    
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        
        with torch.no_grad():
            self.weight.fill_(1.0)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [..., d]
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        variance = torch.clamp(variance, min=self.eps)
        rms = torch.sqrt(variance + self.eps)
        normalized = x / rms
        output = normalized * self.weight
        
        return output


class RMSNormWithClamp(RMSNorm):
    """RMSNorm with additional clamping for extreme numerical stability."""
    
    def __init__(self, d, eps=1e-6, clamp_value=10.0):
        super().__init__(d, eps)
        self.clamp_value = clamp_value
    
    def forward(self, x):
        # First clamp input
        x = torch.clamp(x, -self.clamp_value, self.clamp_value)
        # Call parent forward
        output = super().forward(x)
        # Clamp output again
        output = torch.clamp(output, -self.clamp_value, self.clamp_value)
        
        return output