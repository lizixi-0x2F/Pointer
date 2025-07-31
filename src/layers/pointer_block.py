import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

try:
    from src.layers.alibi import AliBiPositionalEmbedding, apply_alibi_bias
except ImportError:
    try:
        from layers.alibi import AliBiPositionalEmbedding, apply_alibi_bias
    except ImportError:
        from .alibi import AliBiPositionalEmbedding, apply_alibi_bias


def batched_gather(src, idx):
    """Gather values from src using indices in idx.
    
    Args:
        src (torch.Tensor): Source tensor [B, N, d]
        idx (torch.Tensor): Indices tensor [B, N, k]
        
    Returns:
        torch.Tensor: Gathered tensor [B, N, k, d]
    """
    B, N, d = src.shape
    _, _, k = idx.shape
    
    # Expand src to allow gathering
    src_expanded = src.unsqueeze(2).expand(B, N, N, d)  # [B, N, N, d]
    
    # Expand idx to match the last dimension
    idx_expanded = idx.unsqueeze(-1).expand(B, N, k, d)  # [B, N, k, d]
    
    # Gather values - clamp indices to valid range
    idx_clamped = torch.clamp(idx_expanded, 0, N-1)
    gathered = torch.gather(src_expanded, dim=2, index=idx_clamped)
    
    return gathered


class PointerBlock(nn.Module):
    """Pointer Block that generates sparse address distributions and aggregates neighbor vectors.
    
    This is the core component of the Pointer architecture that performs relational routing
    by finding top-k most relevant positions and aggregating their representations.
    
    Args:
        d (int): Hidden dimension
        n_heads (int): Number of query attention heads  
        n_kv_heads (int): Number of key-value heads (for GQA support)
        top_k (int): Number of top positions to select
        use_value_proj (bool): Whether to use value projection
        use_alibi (bool): Whether to use AliBi positional encoding
        max_seq_len (int): Maximum sequence length for AliBi encoding
    """
    
    def __init__(self, d, n_heads, n_kv_heads=None, top_k=2, use_value_proj=True, use_alibi=True, max_seq_len=4096):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.top_k = top_k
        self.head_dim = d // n_heads  # All heads use same head_dim
        self.use_value_proj = use_value_proj
        self.use_alibi = use_alibi
        
        assert d % n_heads == 0, f"Hidden dim {d} must be divisible by n_heads {n_heads}"
        assert n_heads % self.n_kv_heads == 0, f"n_heads {n_heads} must be divisible by n_kv_heads {self.n_kv_heads}"
        
        self.heads_per_kv_group = n_heads // self.n_kv_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)  # Use head_dim, not kv_head_dim
        if use_value_proj:
            self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
            
        # AliBi positional encoding
        if use_alibi:
            self.alibi = AliBiPositionalEmbedding(n_heads, max_seq_len)
            
        # Output projection
        self.o_proj = nn.Linear(d, d, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for FP16 stability."""
        init_std = 0.02 / math.sqrt(self.d)
        for module in [self.q_proj, self.k_proj, self.o_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if self.use_value_proj:
            nn.init.normal_(self.v_proj.weight, mean=0.0, std=init_std)
    
    def forward(self, h, kv_cache=None):
        """Forward pass of PointerBlock.
        
        Args:
            h (torch.Tensor): Input hidden states [B, N, d]
            kv_cache (Optional): KV cache for inference
            
        Returns:
            Tuple containing:
                - z (torch.Tensor): Output representations [B, N, d]
                - topk_idx (torch.Tensor): Top-k indices [B, N, k]  
                - p (torch.Tensor): Pointer probabilities [B, N, k]
        """
        B, N, d = h.shape
        
        
        # Compute Q, K, V
        q = self.q_proj(h)  # [B, N, d]
        
        
        if kv_cache is None:
            k_src = h
            N_cache = N
        else:
            # Get cached values if available
            if hasattr(kv_cache, 'get') and kv_cache.get('vals') is not None:
                cached_vals = kv_cache.get('vals')
                # Only use non-zero part of cache
                cache_pos = kv_cache.get('pos', 0)
                if cache_pos > 0:
                    k_src = cached_vals[:, :cache_pos]
                else:
                    k_src = h
            else:
                k_src = h
            N_cache = k_src.shape[1]
            
        k = self.k_proj(k_src)  # [B, N_cache, n_kv_heads * kv_head_dim]
        
        
        # Skip if either tensor is empty
        if N == 0 or N_cache == 0:
            # Return zero tensors with correct shapes
            z = torch.zeros_like(h)
            topk_idx = torch.zeros(B, N, self.top_k, dtype=torch.long, device=h.device)
            p = torch.zeros(B, N, self.top_k, device=h.device)
            return z, topk_idx, p
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.n_heads, self.head_dim)  # [B, N, H, d_head]  
        k = k.view(B, N_cache, self.n_kv_heads, self.head_dim)  # [B, N_cache, H_kv, d_head]
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [B, H, N, d_head]
        k = k.transpose(1, 2)  # [B, H_kv, N_cache, d_head]
        
        # Expand k for GQA: repeat each KV head for corresponding Q heads
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.heads_per_kv_group, dim=1)  # [B, H, N_cache, d_head]
        
        # Compute attention scores with numerical stability
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N_cache]
        
        
        # Apply AliBi positional bias
        if self.use_alibi:
            alibi_bias = self.alibi(N, N_cache)  # [H, N, N_cache]
            scores = apply_alibi_bias(scores, alibi_bias)
        
        # Clamp scores to prevent extreme values
        scores = torch.clamp(scores, -10.0, 10.0)
        
        # Average across heads for pointer selection
        scores = scores.mean(dim=1)  # [B, N, N_cache]
        
        
        # Get top-k indices and values
        actual_k = min(self.top_k, N_cache)
        topk_val, topk_idx = torch.topk(scores, actual_k, dim=-1)  # [B, N, actual_k]
        
        
        # Pad if necessary to maintain consistent shape
        if actual_k < self.top_k:
            pad_size = self.top_k - actual_k
            topk_val = torch.cat([
                topk_val, 
                torch.full((B, N, pad_size), float('-inf'), device=topk_val.device)
            ], dim=-1)
            topk_idx = torch.cat([
                topk_idx,
                torch.zeros((B, N, pad_size), dtype=torch.long, device=topk_idx.device)
            ], dim=-1)
        
        topk_val_clamped = torch.clamp(topk_val, -5.0, 5.0)
        p = torch.softmax(topk_val_clamped, dim=-1)  # [B, N, k]
        
        # Get value vectors
        if self.use_value_proj:
            v_src = self.v_proj(k_src)  # [B, N_cache, n_kv_heads * head_dim]
        else:
            # If not using value projection, we need to handle the dimension mismatch
            # For GQA compatibility, we project to the correct KV dimension
            if self.n_kv_heads != self.n_heads:
                # Create a temporary projection for non-projected case
                if not hasattr(self, '_temp_v_proj'):
                    self._temp_v_proj = nn.Linear(self.d, self.n_kv_heads * self.head_dim, bias=False)
                    self._temp_v_proj.to(k_src.device)
                    # Initialize with identity-like mapping
                    with torch.no_grad():
                        nn.init.eye_(self._temp_v_proj.weight[:self.d, :self.d])
                v_src = self._temp_v_proj(k_src)
            else:
                v_src = k_src
            
        # Gather top-k values
        gathered = batched_gather(v_src, topk_idx)  # [B, N, k, kv_dim] where kv_dim = n_kv_heads * head_dim
        
        # If GQA, we need to expand gathered values to full dimension
        if self.n_kv_heads != self.n_heads and gathered.shape[-1] != self.d:
            # gathered is [B, N, k, 320], we need [B, N, k, 960]
            # Repeat each KV group to match Q heads
            B, N, k, kv_dim = gathered.shape
            gathered = gathered.view(B, N, k, self.n_kv_heads, self.head_dim)  # [B, N, k, 5, 64]
            gathered = gathered.repeat_interleave(self.heads_per_kv_group, dim=3)  # [B, N, k, 15, 64] 
            gathered = gathered.view(B, N, k, self.d)  # [B, N, k, 960]
        
        # Weighted aggregation
        z = (p.unsqueeze(-1) * gathered).sum(dim=-2)  # [B, N, d]
        
        
        # Output projection
        z = self.o_proj(z)
        
        
        return z, topk_idx, p