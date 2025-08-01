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


def gather_by_pointer(src, ptr):
    """通过指针收集值 - 每个位置只有一个指针指向另一个位置
    
    Args:
        src (torch.Tensor): Source tensor [B, N, d]
        ptr (torch.Tensor): Pointer indices [B, N] - 每个位置指向一个位置
        
    Returns:
        torch.Tensor: Gathered tensor [B, N, d]
    """
    B, N, d = src.shape
    
    # Clamp indices to valid range
    ptr_clamped = torch.clamp(ptr, 0, N-1)
    
    # Use advanced indexing: each position points to exactly one other position
    batch_idx = torch.arange(B, device=src.device)[:, None]  # [B, 1]
    gathered = src[batch_idx, ptr_clamped]  # [B, N, d]
    
    return gathered


class PointerBlock(nn.Module):
    """
    纯关系建模块 - 专注建模 a-->b 的显式关系
    
    核心设计理念（优化版）：
    1. 每个token直接学习指向哪个token（纯关系建模）
    2. 关系链传递：A→B→C，构成显式思维链
    3. 去除注意力机制，专注关系表示：用N个关系替代N×N注意力
    4. 关系作为一等公民：直接建模-->关系，快速构建思维链
    5. 支持反思门控：利用历史关系链进行推理
    
    Args:
        d (int): Hidden dimension
        n_heads (int): Number of heads (简化为关系头数)
        n_kv_heads (int): Number of key-value heads (for compatibility)
        max_seq_len (int): Maximum sequence length
    """
    
    def __init__(self, d, n_heads, n_kv_heads=None, top_k=1, use_value_proj=True, 
                 use_alibi=False, max_seq_len=4096, addressing_mode='learned'):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d // n_heads  
        self.max_seq_len = max_seq_len
        
        assert d % n_heads == 0, f"Hidden dim {d} must be divisible by n_heads {n_heads}"
        
        self.heads_per_kv_group = n_heads // self.n_kv_heads
        
        # 🎯 核心：纯关系学习网络 - 简化设计
        # 直接学习 a-->b 的关系映射
        self.relation_encoder = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),  # 更稳定的激活函数
            nn.Linear(d // 2, 1)  # 输出关系强度
        )
        
        # 🚀 关系值投影：将源token特征转换为关系传递的信息
        self.value_proj = nn.Linear(d, d, bias=False) if use_value_proj else nn.Identity()
        
        # 🔥 关系传递网络：处理A→B关系中的信息传递
        self.relation_transform = nn.Sequential(
            nn.Linear(d * 2, d),  # 输入：[source_token, target_token]的拼接
            nn.GELU(),
            nn.Linear(d, d)
        )
        
        # 简化输出投影
        self.o_proj = nn.Linear(d, d, bias=False)
        
        # 关闭AliBi以提升速度和纯净度
        self.use_alibi = False
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        init_std = 0.02 / math.sqrt(self.d)
        for module in [self.value_proj, self.o_proj]:
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
    
    def _compute_pure_relations(self, h, prev_idx=None):
        """
        🎯 纯关系建模：直接学习 a-->b 的显式关系
        
        Args:
            h (torch.Tensor): Hidden states [B, N, d]
            prev_idx (Optional[torch.Tensor]): Previous layer's relation chain [B, N]
            
        Returns:
            torch.Tensor: Relation targets [B, N] - 每个token指向的目标token
        """
        B, N, d = h.shape
        device = h.device
        
        # 🎯 核心：直接学习关系映射
        # 每个token学习指向哪个位置
        relation_logits = self.relation_encoder(h).squeeze(-1)  # [B, N]
        
        # 转换为位置索引（更简单直接）
        relation_targets = torch.sigmoid(relation_logits) * (N - 1)
        relation_targets = relation_targets.round().long()
        relation_targets = torch.clamp(relation_targets, 0, N - 1)
        
        # 🚀 关系链继承：基于prev_idx形成思维链
        if prev_idx is not None:
            # 策略：后半部分token更倾向于继承关系链
            chain_threshold = N // 2
            should_chain = torch.arange(N, device=device) >= chain_threshold
            should_chain = should_chain.unsqueeze(0).expand(B, N)
            
            # 关系链传递：A→B, B→C => A→B→C
            prev_idx_clamped = torch.clamp(prev_idx, 0, N - 1)
            relation_targets = torch.where(should_chain, prev_idx_clamped, relation_targets)
        
        return relation_targets
    
    def _pure_relation_aggregation(self, h, relation_targets):
        """
        🔥 纯关系信息聚合：处理 a-->b 中的信息传递
        
        Args:
            h (torch.Tensor): Source hidden states [B, N, d]
            relation_targets (torch.Tensor): Relation targets [B, N]
            
        Returns:
            torch.Tensor: Relation-aggregated features [B, N, d]
        """
        B, N, d = h.shape
        
        # 1. 获取关系目标的特征
        batch_idx = torch.arange(B, device=h.device)[:, None]  # [B, 1]
        target_features = h[batch_idx, relation_targets]  # [B, N, d]
        
        # 2. 关系值投影
        source_values = self.value_proj(h)  # [B, N, d]
        target_values = self.value_proj(target_features)  # [B, N, d]
        
        # 3. 🎯 核心：关系传递网络处理 source→target 的信息流
        # 拼接源和目标特征，学习关系传递
        relation_input = torch.cat([source_values, target_values], dim=-1)  # [B, N, 2d]
        relation_output = self.relation_transform(relation_input)  # [B, N, d]
        
        return relation_output
    
    def forward(self, h, kv_cache=None, prev_idx=None, return_full_scores=False):
        """
        纯关系建模的前向传播 - 专注 a-->b 显式关系
        
        Args:
            h (torch.Tensor): Input hidden states [B, N, d]
            kv_cache (Optional): KV cache for inference (简化处理)
            prev_idx (Optional[torch.Tensor]): Previous layer relation targets [B, N] for chaining
            return_full_scores (bool): Whether to return full position scores (兼容性)
            
        Returns:
            Tuple containing:
                - z (torch.Tensor): Output representations [B, N, d]
                - relation_targets (torch.Tensor): Relation targets [B, N] - each token points to one target
                - relation_strength (torch.Tensor): Relation strength [B, N] - strength of each relation
                - full_scores (Optional): Full scores if requested (for compatibility)
        """
        B, N, d = h.shape
        
        # 处理缓存（简化版本，专注关系建模）
        if kv_cache is None:
            h_src = h
            N_cache = N
        else:
            # 简化的缓存处理
            if hasattr(kv_cache, 'get') and kv_cache.get('vals') is not None:
                cached_vals = kv_cache.get('vals')
                cache_pos = kv_cache.get('pos', 0)
                if cache_pos > 0:
                    h_src = cached_vals[:, :cache_pos]
                else:
                    h_src = h
            else:
                h_src = h
            N_cache = h_src.shape[1]
        
        # 边界检查
        if N == 0 or N_cache == 0:
            z = torch.zeros_like(h)
            relation_targets = torch.zeros(B, N, dtype=torch.long, device=h.device)
            relation_strength = torch.zeros(B, N, device=h.device)
            if return_full_scores:
                full_scores = torch.zeros(B, N, N_cache, device=h.device)
                return z, relation_targets, relation_strength, full_scores
            else:
                return z, relation_targets, relation_strength
        
        # 🎯 步骤1：学习纯关系 - 每个token学习指向哪个token
        relation_targets = self._compute_pure_relations(h, prev_idx)  # [B, N]
        
        # 🔥 步骤2：关系信息聚合 - 处理 a-->b 的信息传递
        relation_output = self._pure_relation_aggregation(h, relation_targets)  # [B, N, d]
        
        # 🚀 步骤3：输出投影
        z = self.o_proj(relation_output)
        
        # 计算关系强度（用于兼容性和分析）
        # 使用关系编码器的输出作为强度指标
        relation_logits = self.relation_encoder(h).squeeze(-1)  # [B, N]
        relation_strength = torch.sigmoid(relation_logits)  # [B, N] 归一化到[0,1]
        
        if return_full_scores:
            # 为兼容性创建全分数矩阵（实际上是稀疏的）
            full_scores = torch.zeros(B, N, N_cache, device=h.device)
            
            # 在对应的关系目标位置设置强度
            batch_idx = torch.arange(B, device=h.device)[:, None]
            seq_idx = torch.arange(N, device=h.device)[None, :]
            relation_targets_clamped = torch.clamp(relation_targets, 0, N_cache - 1)
            full_scores[batch_idx, seq_idx, relation_targets_clamped] = relation_strength
            
            return z, relation_targets, relation_strength, full_scores
        else:
            return z, relation_targets, relation_strength
