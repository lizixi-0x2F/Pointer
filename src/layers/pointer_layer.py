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
    
    def __init__(self, d, n_heads, layer_idx=0, n_kv_heads=None, d_ff=None, dropout=0.0,
                 use_value_proj=True, use_alibi=True, max_seq_len=4096, reflection_config=None):
        super().__init__()
        self.d = d
        self.layer_idx = layer_idx
        
        # 移除自动分叉功能 - 专注纯指针关系建模
        
        # 🧠 全局反思机制：每个层都默认具备
        self.reflection_config = reflection_config or {}
        # 可学习的全局反思参数
        self.reflection_history_weight_decay = nn.Parameter(torch.tensor(0.8))  # 可学习的指数衰减
        self.reflection_current_weight = nn.Parameter(torch.tensor(0.7))  # 可学习的当前状态权重
        self.reflection_history_weight = nn.Parameter(torch.tensor(0.3))  # 可学习的历史状态权重
        
        # 可学习的反思强度和门控
        self.global_reflection_gate = nn.Linear(d, 1)  # 学习何时启用反思
        self.reflection_intensity = nn.Parameter(torch.tensor(0.1))  # 可学习的反思强度
        self.reflection_norm = RMSNorm(d)
        self.reflection_proj = nn.Linear(d, d, bias=False)
        # 移除backtrack_layers限制，改为使用全部历史层
        
        # DeepSeek-style RMSNorm (Pre-norm architecture)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        
        # Pointer block (使用可学习的分叉参数)
        self.pointer_block = PointerBlock(
            d=d,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_value_proj=use_value_proj,
            use_alibi=use_alibi,
            max_seq_len=max_seq_len,
            # 使用简化的指针配置
        )
        
        # Learnable gate for pointer output (preserves original design)
        self.gate = nn.Parameter(torch.ones(d))
        
        # 移除特定层反思配置 - 现在所有层都有反思能力
        # if self.is_reflection_layer: 这个条件判断已经不需要了
        
        # Llama-style SwiGLU FFN
        self.ffn = LlamaMLP(
            hidden_size=d,
            intermediate_size=d_ff or int(8 * d / 3),
            hidden_act="silu"
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"PointerLayer {layer_idx} initialized: d={d}, n_heads={n_heads}, "
              f"bidirectional_multihead=True, global_reflection=True")
    
    def _apply_global_reflection_mechanism(self, h, layer_history=None, pointer_history=None):
        """Apply learnable global reflection mechanism.
        
        全局反思机制 - 每个层都具备反思能力，通过可学习参数决定：
        1. 是否启用反思 (learnable gate)
        2. 反思强度 (learnable intensity)
        3. 全局历史状态的融合
        
        Args:
            h (torch.Tensor): Current hidden states [B, N, d]
            layer_history (List[torch.Tensor]): History of hidden states from all previous layers
            pointer_history (List[torch.Tensor]): History of pointer indices from all previous layers
            
        Returns:
            torch.Tensor: Reflection-enhanced hidden states [B, N, d]
        """
        if layer_history is None or len(layer_history) == 0:
            return h
        
        B, N, d = h.shape
        
        # 🧠 可学习的反思门控：决定是否启用反思
        reflection_gate = torch.sigmoid(self.global_reflection_gate(h))  # [B, N, 1]
        
        # 🌐 全局历史状态聚合
        global_context = self._compute_global_context(h, layer_history, pointer_history)
        
        # 🔥 可学习的反思特征生成
        reflection_features = self.reflection_proj(self.reflection_norm(global_context))
        
        # 🎯 动态反思强度调制
        dynamic_intensity = torch.sigmoid(self.reflection_intensity) * reflection_gate
        
        # Apply reflection
        reflected_h = h + dynamic_intensity * reflection_features
        
        # 保存反思特征用于分析
        self.last_reflection_features = reflection_features.clone()
        self.last_reflection_gate = reflection_gate.clone()
        
        return reflected_h
    
    def _compute_global_context(self, h, layer_history, pointer_history):
        """计算全局上下文 - 融合所有历史层的信息
        
        Args:
            h: 当前隐状态 [B, N, d]
            layer_history: 历史层状态列表
            pointer_history: 历史指针列表
            
        Returns:
            global_context: 全局上下文 [B, N, d]
        """
        B, N, d = h.shape
        
        if not layer_history:
            return h
        
        # 简单而有效的全局聚合：加权平均历史状态
        # 使用可学习的权重参数
        weighted_states = []
        total_weight = 0
        
        for i, hist_state in enumerate(layer_history):
            # 距离权重：使用可学习的衰减参数
            weight = torch.pow(self.reflection_history_weight_decay, i)  # 可学习的指数衰减
            weighted_states.append(weight * hist_state)
            total_weight += weight
        
        if total_weight > 0:
            global_history = torch.stack(weighted_states, dim=0).sum(dim=0) / total_weight
        else:
            global_history = layer_history[-1]  # fallback
        
        # 融合当前状态和全局历史 - 使用可学习的权重
        alpha = torch.sigmoid(self.reflection_current_weight)  # 当前状态权重
        beta = torch.sigmoid(self.reflection_history_weight)   # 历史状态权重
        # 归一化权重
        total_weight = alpha + beta
        alpha = alpha / total_weight
        beta = beta / total_weight
        global_context = alpha * h + beta * global_history
        
        return global_context
    
    def forward(self, h, kv_cache=None, prev_idx=None, layer_history=None, pointer_history=None, return_full_scores=False):
        """DeepSeek-style forward pass with global reflection.
        
        Args:
            h (torch.Tensor): Input hidden states [B, N, d]
            kv_cache (Optional): KV cache for inference
            prev_idx (Optional[torch.Tensor]): Previous layer's pointer indices for chaining
            layer_history (Optional[List[torch.Tensor]]): History of hidden states for global reflection
            pointer_history (Optional[List[torch.Tensor]]): History of pointer indices for global reflection
            return_full_scores (bool): Whether to return full position scores
            
        Returns:
            Tuple containing:
                - h (torch.Tensor): Output hidden states [B, N, d]
                - idx (torch.Tensor): Current layer's pointer indices [B, N] 
                - p (torch.Tensor): Pointer probabilities [B, N]
                - full_scores (Optional[torch.Tensor]): Full position scores if requested
        """
        # 🧠 全局反思机制：每个层都默认具备，通过可学习参数控制
        h = self._apply_global_reflection_mechanism(h, layer_history, pointer_history)
        
        # --- Pointer part (Pre-norm) ---
        residual = h
        
        # Pre-norm: normalize then compute
        h_norm = self.norm1(h)
        
        # 使用简化的指针机制，专注关系建模
        pointer_result = self.pointer_block(h_norm, kv_cache, prev_idx=prev_idx, return_full_scores=return_full_scores)
        
        # Always ensure we have the right number of values
        if return_full_scores:
            if len(pointer_result) == 4:
                z, idx, p, full_scores = pointer_result
            else:
                z, idx, p = pointer_result
                full_scores = None  # Fallback if PointerBlock doesn't return full_scores
        else:
            if len(pointer_result) == 4:
                z, idx, p, full_scores = pointer_result
                full_scores = None  # We don't need it
            else:
                z, idx, p = pointer_result
            full_scores = None
        
        # Apply gate and residual connection
        h = residual + self.gate * self.dropout(z)
        
        # --- SwiGLU FFN part (Pre-norm) ---
        residual = h
        
        # Pre-norm: normalize then compute
        h_norm = self.norm2(h)
        
        # Apply SwiGLU FFN
        ffn_out = self.ffn(h_norm)
        
        # Residual connection
        h = residual + self.dropout(ffn_out)
        
        if return_full_scores:
            return h, idx, p, full_scores
        else:
            return h, idx, p
