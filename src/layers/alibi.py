"""
AliBi (Attention with Linear Biases) Positional Encoding
More stable than rotary embedding, especially suitable for FP16 training
"""

import torch
import torch.nn as nn
import math


def get_alibi_slopes(n_heads):
    """Generate AliBi slopes
    
    Args:
        n_heads (int): Number of attention heads
        
    Returns:
        torch.Tensor: Slope vector [n_heads]
    """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(n_heads))
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        slopes += get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n_heads-closest_power_of_2]
        return torch.tensor(slopes)


class AliBiPositionalEmbedding(nn.Module):
    """AliBi positional encoding - numerically stable positional encoding scheme"""
    
    def __init__(self, n_heads, max_seq_len=4096):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        slopes = get_alibi_slopes(n_heads)
        self.register_buffer('slopes', slopes)
        # 不要预先创建position_bias，而是在forward时动态计算
    
    def _create_position_bias(self, seq_len_q, seq_len_k):
        """Create position bias matrix dynamically"""
        device = self.slopes.device
        
        # 创建位置矩阵，但不存储为buffer
        pos_q = torch.arange(seq_len_q, device=device).unsqueeze(1)
        pos_k = torch.arange(seq_len_k, device=device).unsqueeze(0)
        pos = pos_q - pos_k  # [seq_len_q, seq_len_k]
        
        # 只保留下三角部分（因果mask）
        mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=device, dtype=torch.bool))
        pos = torch.where(mask, pos, torch.tensor(0, device=device))
        
        # 应用斜率
        bias = pos.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)  # [n_heads, seq_len_q, seq_len_k]
        bias = torch.clamp(bias, -50.0, 0.0)
        
        return bias
    
    def forward(self, seq_len_q, seq_len_k):
        """
        Args:
            seq_len_q (int): Query sequence length
            seq_len_k (int): Key sequence length
            
        Returns:
            torch.Tensor: AliBi bias [n_heads, seq_len_q, seq_len_k]
        """
        return self._create_position_bias(seq_len_q, seq_len_k)


def apply_alibi_bias(scores, alibi_bias):
    """Apply AliBi bias to attention scores
    
    Args:
        scores (torch.Tensor): attention scores [B, H, N_q, N_k]
        alibi_bias (torch.Tensor): AliBi bias [H, N_q, N_k]
        
    Returns:
        torch.Tensor: Biased scores [B, H, N_q, N_k]
    """
    if alibi_bias is None:
        return scores
    alibi_bias = torch.clamp(alibi_bias, -50.0, 0.0)
    biased_scores = scores + alibi_bias.unsqueeze(0)
    biased_scores = torch.clamp(biased_scores, -50.0, 50.0)
    return biased_scores