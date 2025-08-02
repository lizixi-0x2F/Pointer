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
                 use_value_proj=True, use_alibi=True, max_seq_len=4096, reflection_config=None,
                 dynamic_threshold=0.3, max_branches=3):
        super().__init__()
        self.d = d
        self.layer_idx = layer_idx
        
        # Reflection configuration
        self.reflection_config = reflection_config or {}
        self.is_reflection_layer = layer_idx in self.reflection_config.get('reflection_layers', [])
        # 移除backtrack_layers限制，改为使用全部历史层
        
        # DeepSeek-style RMSNorm (Pre-norm architecture)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        
        # Pointer block (using AliBi)
        self.pointer_block = PointerBlock(
            d=d,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_value_proj=use_value_proj,
            use_alibi=use_alibi,
            max_seq_len=max_seq_len,
            dynamic_threshold=dynamic_threshold,
            max_branches=max_branches
        )
        
        # Learnable gate for pointer output (preserves original design)
        self.gate = nn.Parameter(torch.ones(d))
        
        # Reflection gating mechanism
        if self.is_reflection_layer:
            # Reflection-specific components
            gate_init = self.reflection_config.get('reflection_gate_init', 0.1)
            self.reflection_gate = nn.Parameter(
                torch.full((d,), gate_init)
            )
            self.reflection_norm = RMSNorm(d)
            self.reflection_proj = nn.Linear(d, d, bias=False)
            
            print(f"PointerLayer {layer_idx} initialized with REFLECTION support (gate_init={gate_init})")
        
        # Llama-style SwiGLU FFN
        self.ffn = LlamaMLP(
            hidden_size=d,
            intermediate_size=d_ff or int(8 * d / 3),
            hidden_act="silu"
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"PointerLayer {layer_idx} initialized (DynamicPointer): d={d}, n_heads={n_heads}, branches={max_branches}")
    
    def _apply_reflection_mechanism(self, h, layer_history=None, pointer_history=None, return_gates=False):
        """Apply enhanced reflection mechanism for pure relation modeling.
        
        在纯关系建模范式下，反思机制专注于关系链的学习和传承：
        1. 分析历史关系模式：哪些关系是一致的、有效的
        2. 关系链反思：A→B→C 链条的连贯性
        3. 动态门控：基于关系质量调整反思强度
        
        Args:
            h (torch.Tensor): Current hidden states [B, N, d]
            layer_history (List[torch.Tensor]): History of hidden states from previous layers
            pointer_history (List[torch.Tensor]): History of relation targets from previous layers
            return_gates (bool): Whether to return gate values
            
        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: 
                Reflection-enhanced hidden states [B, N, d]
                If return_gates=True, also returns gate values [B, N]
        """
        if not self.is_reflection_layer or layer_history is None:
            # 保存原始特征用于损失计算
            self.last_reflection_features = h.clone() if self.is_reflection_layer else None
            return h
        
        # 使用全部历史层进行全局回溯
        relevant_history = layer_history[:] if layer_history else []
        relevant_pointers = pointer_history[:] if pointer_history and isinstance(pointer_history, list) else []
        
        # 构建全局关系图
        global_relation_graph = self._build_global_relation_graph(relevant_pointers, h.device)
        
        if not relevant_history:
            self.last_reflection_features = h.clone()
            return h
        
        # 🎯 新设计：纯关系导向的反思机制
        B, N, d = h.shape
        
        # 1. 关系链一致性分析
        relation_consistency = self._analyze_relation_chains(relevant_pointers, B, N, h.device)
        
        # 2. 基于关系链的历史状态加权聚合
        weighted_history = self._relation_weighted_history_aggregation(
            relevant_history, relevant_pointers, relation_consistency, B, N, d
        )
        
        # 3. 关系导向的反思特征生成
        relation_context = self._compute_relation_context(h, weighted_history, relation_consistency)
        reflection_features = self.reflection_proj(self.reflection_norm(relation_context))
        
        # 4. 🔥 动态关系门控：基于关系质量调整反思强度
        relation_quality = self._compute_relation_quality(relevant_pointers, relation_consistency)
        dynamic_gate = self.reflection_gate * relation_quality.unsqueeze(-1)
        
        # Apply dynamic reflection gate
        reflected_h = h + dynamic_gate * reflection_features
        
        # 保存反思特征用于损失计算
        self.last_reflection_features = reflection_features.clone()
        
        if return_gates:
            # 返回门控值 (取动态门控的平均值)
            gate_values = dynamic_gate.mean(dim=-1)  # [B, N]
            return reflected_h, gate_values
        return reflected_h
    
    def _build_global_relation_graph(self, pointer_history, device):
        """构建全局关系图"""
        if not pointer_history:
            return None
            
        B, N = pointer_history[0].shape
        # 初始化关系图 [B, N, N]
        relation_graph = torch.zeros(B, N, N, device=device)
        
        # 统计所有历史层的关系
        for ptr in pointer_history:
            batch_idx = torch.arange(B, device=device)[:, None, None]
            seq_idx = torch.arange(N, device=device)[None, :, None]
            ptr_clamped = torch.clamp(ptr, 0, N-1)[..., None]
            
            # 在关系图中累加关系出现次数
            relation_graph[batch_idx, seq_idx, ptr_clamped] += 1
        
        # 归一化
        relation_graph = relation_graph / len(pointer_history)
        return relation_graph

    def _analyze_relation_chains(self, pointer_history, B, N, device):
        """分析关系链的一致性和质量(全局版本)"""
        if not pointer_history:
            return torch.ones(B, N, device=device)
        
        # 使用全局关系图分析
        relation_graph = self._build_global_relation_graph(pointer_history, device)
        
        # 计算每个位置的全局关系稳定性
        chain_stability = torch.zeros(B, N, device=device)
        if relation_graph is not None:
            # 稳定性 = 主要关系占比
            chain_stability = relation_graph.max(dim=-1).values
        
        # 加入位置偏好：前面的token更容易形成稳定关系
        position_bias = torch.linspace(1.0, 0.5, N, device=device).unsqueeze(0).expand(B, N)
        chain_stability = chain_stability * position_bias + 0.1  # 最小基础值
        
        return chain_stability
    
    def _relation_weighted_history_aggregation(self, history_states, pointer_history, consistency, B, N, d):
        """基于关系一致性的历史状态加权聚合"""
        if not history_states:
            return torch.zeros(B, N, d, device=consistency.device)
        
        # 将历史状态按关系一致性加权
        weighted_states = []
        
        for i, state in enumerate(history_states):
            # 使用对应的关系一致性作为权重
            if i < len(pointer_history) and pointer_history:
                # 基于关系稳定性调整历史状态的重要性
                weight = consistency + 0.1  # 确保最小权重
            else:
                weight = torch.ones_like(consistency) * 0.5  # 默认权重
            
            weighted_state = state * weight.unsqueeze(-1)
            weighted_states.append(weighted_state)
        
        # 聚合加权状态
        aggregated = torch.stack(weighted_states, dim=0).mean(dim=0)  # [B, N, d]
        return aggregated
    
    def _compute_relation_context(self, current_h, historical_h, consistency):
        """计算关系上下文：当前状态与历史关系的融合"""
        # 简单但有效的上下文计算
        B, N, d = current_h.shape
        
        # 基于关系一致性混合当前和历史状态
        alpha = consistency.unsqueeze(-1)  # [B, N, 1]
        beta = 1.0 - alpha
        
        # 上下文 = α * 历史 + β * 现在
        relation_context = alpha * historical_h + beta * current_h
        
        return relation_context
    
    def _compute_relation_quality(self, pointer_history, consistency):
        """计算关系质量分数，用于动态门控(全局版本)"""
        if not pointer_history:
            return torch.ones_like(consistency)
        
        B, N = consistency.shape
        device = consistency.device
        
        # 使用全局关系图计算质量
        relation_graph = self._build_global_relation_graph(pointer_history, device)
        if relation_graph is None:
            return consistency
        
        # 质量基于：
        # 1) 主要关系占比 (consistency)
        # 2) 关系多样性 (1 - 主要关系占比)
        # 3) 非自指程度
        quality_score = consistency.clone()
        
        # 计算全局多样性 (1 - 主要关系占比)
        main_relation_ratio = relation_graph.max(dim=-1).values
        diversity = 1.0 - main_relation_ratio
        quality_score = quality_score * (0.7 + 0.3 * diversity)  # 适当奖励多样性
        
        # 惩罚自指关系 (使用全局关系图中的自指比例)
        position_indices = torch.arange(N, device=device)[None, :, None]
        self_pointing = relation_graph.gather(-1, position_indices).squeeze(-1)
        quality_score = quality_score * (1.0 - 0.5 * self_pointing)  # 更强惩罚自指
        
        return torch.clamp(quality_score, 0.1, 1.0)
    
    def forward(self, h, kv_cache=None, prev_idx=None, layer_history=None, pointer_history=None, return_full_scores=False):
        """DeepSeek-style forward pass with Pre-norm architecture and reflection support.
        
        Args:
            h (torch.Tensor): Input hidden states [B, N, d]
            kv_cache (Optional): KV cache for inference
            prev_idx (Optional[torch.Tensor]): Previous layer's pointer indices for chaining
            layer_history (Optional[List[torch.Tensor]]): History of hidden states for reflection
            pointer_history (Optional[List[torch.Tensor]]): History of pointer indices for relationship reflection
            return_full_scores (bool): Whether to return full position scores
            
        Returns:
            Tuple containing:
                - h (torch.Tensor): Output hidden states [B, N, d]
                - idx (torch.Tensor): Current layer's pointer indices [B, N] 
                - p (torch.Tensor): Pointer probabilities [B, N]
                - full_scores (Optional[torch.Tensor]): Full position scores if requested
        """
        # Apply reflection mechanism if this is a reflection layer (改进版)
        if self.is_reflection_layer:
            h = self._apply_reflection_mechanism(h, layer_history, pointer_history)
        
        # --- Pointer part (Pre-norm) ---
        residual = h
        
        # Pre-norm: normalize then compute
        h_norm = self.norm1(h)
        
        # Apply pointer block (now returns single pointer per position)
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
        
        # Note: pointer chaining is now handled inside PointerBlock._compute_pointer_relationships
        # idx is now [B, N] instead of [B, N, k] - each position points to exactly one other position
        
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
