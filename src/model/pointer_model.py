import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Any

try:
    from src.layers import TokenEmbedding, PointerLayer
    from src.layers.rmsnorm import RMSNorm
except ImportError:
    # Alternative import paths
    try:
        from layers import TokenEmbedding, PointerLayer
        from layers.rmsnorm import RMSNorm
    except ImportError:
        # Final fallback - absolute imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from layers import TokenEmbedding, PointerLayer
        from layers.rmsnorm import RMSNorm


class PointerCache:
    """Cache for storing key-value pairs and indices during inference.
    
    Args:
        max_batch_size (int): Maximum batch size
        max_seq_len (int): Maximum sequence length  
        n_heads (int): Number of attention heads
        head_dim (int): Dimension per head
        device (torch.device): Device to store cache
        dtype (torch.dtype): Data type for cache
    """
    
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, device, dtype):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        # Initialize cache tensors
        cache_shape = (max_batch_size, max_seq_len, n_heads * head_dim)
        self.vals = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.pos = 0
        
    def append(self, k):
        """Append new key values to cache.
        
        Args:
            k (torch.Tensor): New key values [B, N, d]
        """
        B, N, d = k.shape
        end_pos = self.pos + N
        
        # Ensure cache has enough space
        if end_pos > self.vals.size(1):
            # Expand cache if needed
            new_size = max(end_pos, self.vals.size(1) * 2)
            new_vals = torch.zeros(self.vals.size(0), new_size, self.vals.size(2), 
                                 dtype=self.vals.dtype, device=self.vals.device)
            new_vals[:, :self.pos] = self.vals[:, :self.pos]
            self.vals = new_vals
        
        # Ensure k has the right shape for caching
        if k.dim() == 2:  # [N, d] -> [1, N, d]
            k = k.unsqueeze(0)
        
        self.vals[:B, self.pos:end_pos] = k
        self.pos = end_pos
    
    def get(self, key, default=None):
        """Get cache value by key (for compatibility)."""
        if key == 'vals':
            return self.vals[:, :self.pos]
        elif key == 'pos':
            return self.pos
        return default
    
    def reset(self):
        """Reset cache position."""
        self.pos = 0


class PointerDecoder(nn.Module):
    """DeepSeek-style Pointer Decoder model implementing the Pointer architecture.
    
    Architecture: TokenEmbed → Multi-layer PointerLayer → RMSNorm → LM Head
    Uses DeepSeek components: RMSNorm + SwiGLU FFN + Pre-norm architecture
    Supports left-to-right decoding with KV caching for inference.
    
    Args:
        vocab_size (int): Vocabulary size
        d (int): Hidden dimension
        n_layers (int): Number of layers
        n_heads (int): Number of attention heads
        top_k (int): Top-k for pointer selection
        d_ff (int): Feed-forward hidden dimension (if None, auto-calculated)
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout rate
        tie_embeddings (bool): Whether to tie input/output embeddings
    """
    
    def __init__(self, vocab_size, d, n_layers, n_heads, n_kv_heads=None, top_k=2, d_ff=None, 
                 max_seq_len=4096, dropout=0.0, tie_embeddings=True, fp16=False, reflection_config=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d // n_heads
        self.max_seq_len = max_seq_len
        self.fp16 = fp16
        self.reflection_config = reflection_config or {}
        
        if d_ff is None:
            d_ff = int(8 * d / 3)
        self.d_ff = d_ff
        
        # Token embedding
        self.embed = TokenEmbedding(vocab_size, d, dropout)
        
        # Pointer layers with reflection support
        self.layers = nn.ModuleList([
            PointerLayer(
                d=d,
                n_heads=n_heads,
                layer_idx=layer_idx,  # Pass layer index for reflection control
                n_kv_heads=self.n_kv_heads,
                top_k=top_k,
                d_ff=self.d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len,
                reflection_config=self.reflection_config  # Pass reflection config
            ) for layer_idx in range(n_layers)
        ])
        
        # Final RMSNorm
        self.ln_f = RMSNorm(d)
        
        # Language modeling head
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Add gradient checkpointing support
        self.gradient_checkpointing = False
        
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"PointerDecoder initialized: {total_params:.1f}M parameters")
        print(f"Config: d={d}, layers={n_layers}, heads={n_heads}, kv_heads={self.n_kv_heads}, top_k={top_k}")
        
        # Note: Don't convert to fp16 here, let the trainer handle it
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling for numerical stability."""
        if isinstance(module, nn.Linear):
            init_std = 0.02 / math.sqrt(module.weight.size(-1))
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init_std = 0.02 / math.sqrt(module.weight.size(-1))
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        self.gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self.gradient_checkpointing = False
    
    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, cache=None, output_hiddens=False):
        """Forward pass for training.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [B, N]
            attention_mask (Optional[torch.Tensor]): Attention mask [B, N]
            labels (Optional[torch.Tensor]): Target labels for computing loss [B, N]
            use_cache (bool): Whether to use/return cache
            cache (Optional[List]): Cache from previous forward passes
            output_hiddens (bool): Whether to return hidden states for distillation
            
        Returns:
            Dict containing logits, loss (if labels provided), and optionally cache, hiddens
        """
        B, N = input_ids.shape
        device = input_ids.device
        
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        h = self.embed(input_ids)
        
        # Initialize cache if needed (only for inference, not training)
        if use_cache and cache is None and not self.training:
            cache = [
                PointerCache(B, self.max_seq_len, self.n_kv_heads, self.d // self.n_heads, device, h.dtype)
                for _ in range(self.n_layers)
            ]
        
        # Apply pointer layers with reflection support
        idx = None
        all_pointer_probs = []
        all_hiddens = []  # For distillation
        layer_history = []  # For reflection mechanism
        reflection_info = {
            'layer_history': [],  # 每个反思层的历史状态
            'reflection_gates': [],  # 反思门控值
            'reflection_features': [],  # 反思特征
            'reflection_layers': []  # 反思层索引
        }
        
        for i, layer in enumerate(self.layers):
            # Store hidden states for distillation
            if output_hiddens:
                all_hiddens.append(h.clone())
                
            # Only use cache during inference, not training
            layer_cache = cache[i] if (cache and not self.training) else None
            
            # Apply gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                h, idx, p = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    h, layer_cache, idx, layer_history,
                    use_reentrant=False
                )
            else:
                h, idx, p = layer(h, kv_cache=layer_cache, prev_idx=idx, layer_history=layer_history)
            
            # 如果是反思层，收集反思信息
            if hasattr(layer, 'is_reflection_layer') and layer.is_reflection_layer:
                reflection_info['reflection_layers'].append(i)
                reflection_info['layer_history'].append(layer_history.copy() if layer_history else [])
                
                # 获取反思门控值和特征
                if hasattr(layer, 'reflection_gate') and hasattr(layer, 'last_reflection_features'):
                    gate_value = layer.reflection_gate.data.mean()  # 简化的门控值
                    reflection_info['reflection_gates'].append(gate_value.unsqueeze(0).expand(h.shape[0], h.shape[1]))
                    
                    if hasattr(layer, 'last_reflection_features') and layer.last_reflection_features is not None:
                        reflection_info['reflection_features'].append(layer.last_reflection_features)
                    else:
                        reflection_info['reflection_features'].append(h.clone())
                else:
                    # Fallback: 使用当前隐状态作为反思特征
                    reflection_info['reflection_gates'].append(torch.ones_like(h[:, :, 0]))
                    reflection_info['reflection_features'].append(h.clone())
            
            # Store current layer's hidden states for reflection
            layer_history.append(h.detach().clone())  # Detach to avoid gradients through history
            
            # Limit history size to prevent memory explosion
            max_history = self.reflection_config.get('pointer_backtrack_layers', 8)
            if len(layer_history) > max_history:
                layer_history.pop(0)  # Remove oldest
            
            all_pointer_probs.append(p)
            
            # Update cache only during inference
            if use_cache and cache and not self.training and layer_cache is not None:
                layer_cache.append(h)
        
        # Final layer norm and projection
        h = self.ln_f(h)
        
            
        logits = self.lm_head(h)  # [B, N, vocab_size]
        
        
        result = {'logits': logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            # Shift labels for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy calculation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
            result['loss'] = loss
        
        if use_cache and not self.training:
            result['cache'] = cache
            result['pointer_probs'] = all_pointer_probs
        elif self.training:
            # During training, always return pointer probs for distillation
            result['pointer_probs'] = all_pointer_probs
            if output_hiddens:
                result['hiddens'] = all_hiddens
            
            # 返回反思信息用于反思损失计算
            if reflection_info['reflection_layers']:
                result['reflection_outputs'] = reflection_info
            
        return result
    
    @torch.no_grad()
    def generate_step(self, token, cache):
        """Single generation step for inference.
        
        Args:
            token (torch.Tensor): Current token [B, 1]
            cache (List[PointerCache]): KV cache
            
        Returns:
            Tuple of (logits, updated_cache)
        """
        # Embed token
        h = self.embed(token)  # [B, 1, d]
        
        # Apply layers with caching
        idx = None
        for i, layer in enumerate(self.layers):
            h, idx, _ = layer(h, kv_cache=cache[i], prev_idx=idx)
            cache[i].append(h)
        
        # Final projection
        logits = self.lm_head(self.ln_f(h))  # [B, 1, vocab_size]
        
        return logits, cache
    
    def get_pointer_alignments(self, input_ids):
        """Get pointer alignments for analysis/visualization.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [B, N]
            
        Returns:
            List of pointer probabilities for each layer
        """
        with torch.no_grad():
            result = self.forward(input_ids, use_cache=True)
            return result.get('pointer_probs', [])