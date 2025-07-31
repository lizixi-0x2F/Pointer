import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from src.layers.pointer_block import PointerBlock
    from src.layers.llama_mlp import LlamaMLP
    from src.layers.rmsnorm import RMSNorm
except ImportError:
    # Alternative import paths
    try:
        from layers.pointer_block import PointerBlock
        from layers.llama_mlp import LlamaMLP
        from layers.rmsnorm import RMSNorm
    except ImportError:
        # Relative imports
        from .pointer_block import PointerBlock
        from .llama_mlp import LlamaMLP
        from .rmsnorm import RMSNorm


class PointerLayer(nn.Module):
    """DeepSeek-style Pointer Layer with Reflection Mechanism.
    
    Architecture: RMSNorm → PointerBlock → Residual → RMSNorm → SwiGLU → Residual (Pre-norm)
    Key features: 
    - Passes prev_index to form pointer chains across layers
    - Reflection gating mechanism for structured reasoning
    - Pointer backtracking for reflection layers
    
    Args:
        d (int): Hidden dimension
        n_heads (int): Number of attention heads
        layer_idx (int): Current layer index (for reflection control)
        top_k (int): Number of top positions for pointer selection
        d_ff (int): Feed-forward hidden dimension (if None, auto-calculated)
        dropout (float): Dropout rate
        use_value_proj (bool): Whether to use value projection in PointerBlock
        use_alibi (bool): Whether to use AliBi positional encoding
        max_seq_len (int): Maximum sequence length
        reflection_config (dict): Reflection configuration parameters
    """
    
    def __init__(self, d, n_heads, layer_idx=0, n_kv_heads=None, top_k=2, d_ff=None, dropout=0.0, 
                 use_value_proj=True, use_alibi=True, max_seq_len=4096, reflection_config=None):
        super().__init__()
        self.d = d
        self.layer_idx = layer_idx
        
        # Reflection configuration
        self.reflection_config = reflection_config or {}
        self.is_reflection_layer = layer_idx in self.reflection_config.get('reflection_layers', [])
        self.backtrack_layers = self.reflection_config.get('pointer_backtrack_layers', 8)
        
        # DeepSeek-style RMSNorm (Pre-norm architecture)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        
        # Pointer block (using AliBi)
        self.pointer_block = PointerBlock(
            d=d, 
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            top_k=top_k,
            use_value_proj=use_value_proj,
            use_alibi=use_alibi,
            max_seq_len=max_seq_len
        )
        
        # Learnable gate for pointer output (preserves original design)
        self.gate = nn.Parameter(torch.ones(d))
        
        # Reflection gating mechanism
        if self.is_reflection_layer:
            # Reflection-specific components
            self.reflection_gate = nn.Parameter(
                torch.full((d,), self.reflection_config.get('reflection_gate_init', 0.1))
            )
            self.reflection_norm = RMSNorm(d)
            self.reflection_proj = nn.Linear(d, d, bias=False)
            
            print(f"PointerLayer {layer_idx} initialized with REFLECTION support")
        
        # Llama-style SwiGLU FFN
        self.ffn = LlamaMLP(
            hidden_size=d,
            intermediate_size=d_ff or int(8 * d / 3),
            hidden_act="silu"
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"PointerLayer {layer_idx} initialized (DeepSeek style): d={d}, n_heads={n_heads}, top_k={top_k}")
    
    def _apply_reflection_mechanism(self, h, layer_history=None):
        """Apply reflection mechanism by aggregating information from previous layers.
        
        Args:
            h (torch.Tensor): Current hidden states [B, N, d]
            layer_history (List[torch.Tensor]): History of hidden states from previous layers
            
        Returns:
            torch.Tensor: Reflection-enhanced hidden states [B, N, d]
        """
        if not self.is_reflection_layer or layer_history is None:
            # 保存原始特征用于损失计算
            self.last_reflection_features = h.clone() if self.is_reflection_layer else None
            return h
        
        # Get relevant history (last N layers for backtracking)
        relevant_history = layer_history[-self.backtrack_layers:] if len(layer_history) >= self.backtrack_layers else layer_history
        
        if not relevant_history:
            self.last_reflection_features = h.clone()
            return h
        
        # Aggregate historical information
        historical_info = torch.stack(relevant_history, dim=0).mean(dim=0)  # [B, N, d]
        
        # Reflection projection
        reflection_features = self.reflection_proj(self.reflection_norm(historical_info))
        
        # Apply reflection gate
        reflected_h = h + self.reflection_gate * reflection_features
        
        # 保存反思特征用于损失计算
        self.last_reflection_features = reflection_features.clone()
        
        return reflected_h
    
    def forward(self, h, kv_cache=None, prev_idx=None, layer_history=None):
        """DeepSeek-style forward pass with Pre-norm architecture and reflection support.
        
        Args:
            h (torch.Tensor): Input hidden states [B, N, d]
            kv_cache (Optional): KV cache for inference
            prev_idx (Optional[torch.Tensor]): Previous layer's pointer indices for chaining
            layer_history (Optional[List[torch.Tensor]]): History of hidden states for reflection
            
        Returns:
            Tuple containing:
                - h (torch.Tensor): Output hidden states [B, N, d]
                - idx (torch.Tensor): Current layer's pointer indices [B, N, k]
                - p (torch.Tensor): Pointer probabilities [B, N, k]
        """
        # Apply reflection mechanism if this is a reflection layer
        if self.is_reflection_layer and layer_history is not None:
            h = self._apply_reflection_mechanism(h, layer_history)
        
        # --- Pointer part (Pre-norm) ---
        residual = h
        
        # Pre-norm: normalize then compute
        h_norm = self.norm1(h)
        
        # Apply pointer block
        z, idx, p = self.pointer_block(h_norm, kv_cache)
        
        # Apply gate and residual connection
        h = residual + self.gate * self.dropout(z)
        
        # Chain pointer indices if previous indices exist (pointer-of-pointer)
        if prev_idx is not None:
            # Resolve chained pointers: current_idx = prev_idx[idx]
            B, N, k = idx.shape
            prev_k = prev_idx.shape[-1]
            
        # Simple chaining: for each current idx, look up in prev_idx
            # Clamp indices to valid range
            idx_clamped = torch.clamp(idx, 0, prev_k-1)
            
            # Gather from prev_idx using current idx
            idx = torch.gather(
                prev_idx.unsqueeze(2).expand(B, N, k, prev_k), 
                dim=3, 
                index=idx_clamped.unsqueeze(-1)
            ).squeeze(-1)  # [B, N, k]
        
        # --- SwiGLU FFN part (Pre-norm) ---
        residual = h
        
        # Pre-norm: normalize then compute
        h_norm = self.norm2(h)
        
        # Apply SwiGLU FFN
        ffn_out = self.ffn(h_norm)
        
        # Residual connection
        h = residual + self.dropout(ffn_out)
        
        return h, idx, p