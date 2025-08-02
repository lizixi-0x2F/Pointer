import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List

try:
    from src.layers.alibi import AliBiPositionalEmbedding, apply_alibi_bias
except ImportError:
    try:
        from layers.alibi import AliBiPositionalEmbedding, apply_alibi_bias
    except ImportError:
        from .alibi import AliBiPositionalEmbedding, apply_alibi_bias


def gather_by_pointer(src, ptr):
    """通过指针收集值 - 支持单指针和多跳指针链
    
    Args:
        src (torch.Tensor): Source tensor [B, N, d]
        ptr (torch.Tensor | List[torch.Tensor]): 指针索引或指针链
                  [B, N] 或 List[[B, N], ...]
        
    Returns:
        torch.Tensor | List[torch.Tensor]: 收集结果
    """
    if isinstance(ptr, list):
        return [gather_by_pointer(src, p) for p in ptr]
    
    # 原始单指针逻辑
    B, N, d = src.shape
    
    # Clamp indices to valid range
    ptr_clamped = torch.clamp(ptr, 0, N-1)
    
    # 确保ptr_clamped形状正确 [B, N]
    if ptr_clamped.dim() == 3:
        ptr_clamped = ptr_clamped.squeeze(-1)
    
    # 使用广播索引
    batch_idx = torch.arange(B, device=src.device)[:, None].expand(B, N)  # [B, N]
    gathered = src[batch_idx, ptr_clamped]  # [B, N, d]
    
    return gathered


class PointerChain(nn.Module):
    """多跳指针链模块"""
    def __init__(self, d, max_hops=3):
        super().__init__()
        self.max_hops = max_hops
        self.hop_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(max_hops)])
        self.hop_projs = nn.ModuleList([nn.Linear(d, 1) for _ in range(max_hops)])
    
    def forward(self, h, first_hop_ptr):
        """生成多跳指针链
        Args:
            h: [B, N, d] 输入特征
            first_hop_ptr: [B, N] 第一跳指针
        Returns:
            List[[B, N], ...] 多跳指针链
        """
        ptr_chain = [first_hop_ptr]
        for i in range(1, self.max_hops):
            # 获取上一跳特征
            hop_feat = gather_by_pointer(h, ptr_chain[-1])
            # 计算下一跳指针
            next_ptr = self.hop_projs[i](self.hop_norms[i](hop_feat))
            next_ptr = torch.sigmoid(next_ptr) * (h.size(1) - 1)
            ptr_chain.append(next_ptr.round().long())
        return ptr_chain


class BiDirectionalMultiHeadPointer(nn.Module):
    """
    双向多头指针机制 - 支持不同尺度的关系建模
    
    核心设计理念：
    1. 双向指针：前向和后向关系建模
    2. 多头机制：不同头关注不同尺度的关系
    3. 关系融合：整合多个方向和尺度的信息
    
    Args:
        d (int): Hidden dimension
        n_heads (int): Number of relation heads
        max_seq_len (int): Maximum sequence length
    """
    
    def __init__(self, d, n_heads, max_seq_len=4096):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.max_seq_len = max_seq_len
        
        assert d % n_heads == 0, f"Hidden dim {d} must be divisible by n_heads {n_heads}"
        
        # 前向和后向关系编码器
        self.forward_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Linear(d // 2, 1)
            ) for _ in range(n_heads)
        ])
        
        self.backward_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Linear(d // 2, 1)
            ) for _ in range(n_heads)
        ])
        
        # 多头值投影
        self.multi_head_value_proj = nn.ModuleList([
            nn.Linear(d, self.head_dim, bias=False) for _ in range(n_heads)
        ])
        
        # 双向关系融合网络
        self.relation_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * 3, self.head_dim),  # [source, forward_target, backward_target]
                nn.GELU(),
                nn.Linear(self.head_dim, self.head_dim)
            ) for _ in range(n_heads)
        ])
        
        # 多头输出融合
        self.output_proj = nn.Linear(d, d, bias=False)
        
        # 可学习的链式传承参数
        self.chain_threshold_ratio = nn.Parameter(torch.tensor(0.5))  # 可学习的链阈值比例
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        init_std = 0.02 / math.sqrt(self.d)
        for module_list in [self.multi_head_value_proj]:
            for module in module_list:
                if hasattr(module, 'weight'):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
        
        if hasattr(self.output_proj, 'weight'):
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=init_std)
    
    def forward(self, h, prev_idx=None):
        """
        双向多头指针前向传播
        
        Args:
            h (torch.Tensor): Input hidden states [B, N, d]
            prev_idx (Optional[torch.Tensor]): Previous layer pointers [B, N]
            
        Returns:
            Tuple containing:
                - output (torch.Tensor): Output features [B, N, d]
                - forward_pointers (torch.Tensor): Forward pointers [B, N]
                - backward_pointers (torch.Tensor): Backward pointers [B, N]
                - relation_strength (torch.Tensor): Combined relation strength [B, N]
        """
        B, N, d = h.shape
        device = h.device
        
        all_head_outputs = []
        all_forward_pointers = []
        all_backward_pointers = []
        all_strengths = []
        
        for head_idx in range(self.n_heads):
            # 计算前向和后向关系
            forward_logits = self.forward_encoders[head_idx](h).squeeze(-1)  # [B, N]
            backward_logits = self.backward_encoders[head_idx](h).squeeze(-1)  # [B, N]
            
            # 转换为可微分的指针位置 (使用softmax分布而不是硬位置)
            forward_probs = torch.softmax(forward_logits.unsqueeze(-1).expand(-1, -1, N), dim=-1)  # [B, N, N]
            backward_probs = torch.softmax(backward_logits.unsqueeze(-1).expand(-1, -1, N), dim=-1)  # [B, N, N]
            
            # 计算期望位置用于统计（不参与梯度）
            position_range = torch.arange(N, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            forward_targets = torch.sum(forward_probs * position_range, dim=-1).long()  # [B, N] 
            backward_targets = torch.sum(backward_probs * position_range, dim=-1).long()  # [B, N]
            
            # 链式传承（如果有前一层的指针）- 使用可学习的阈值
            if prev_idx is not None:
                chain_threshold = int(torch.sigmoid(self.chain_threshold_ratio) * N)  # 可学习的阈值
                should_chain = torch.arange(N, device=device) >= chain_threshold
                should_chain = should_chain.unsqueeze(0).expand(B, N)
                prev_idx_clamped = torch.clamp(prev_idx, 0, N - 1)
                forward_targets = torch.where(should_chain, prev_idx_clamped, forward_targets)
            
            # 提取多头特征
            head_features = self.multi_head_value_proj[head_idx](h)  # [B, N, head_dim]
            
            # 可微分的双向特征收集 (使用概率加权)
            forward_features = torch.bmm(forward_probs, head_features)  # [B, N, head_dim]
            backward_features = torch.bmm(backward_probs, head_features)  # [B, N, head_dim]
            
            # 三元关系融合：[source, forward_target, backward_target]
            relation_input = torch.cat([head_features, forward_features, backward_features], dim=-1)
            head_output = self.relation_fusion[head_idx](relation_input)  # [B, N, head_dim]
            
            # 计算关系强度 (使用概率分布的集中度)
            forward_strength = 1.0 - torch.sum(forward_probs * torch.log(forward_probs + 1e-8), dim=-1)  # 熵的负值
            backward_strength = 1.0 - torch.sum(backward_probs * torch.log(backward_probs + 1e-8), dim=-1)
            combined_strength = (forward_strength + backward_strength) / 2
            
            all_head_outputs.append(head_output)
            all_forward_pointers.append(forward_targets)
            all_backward_pointers.append(backward_targets)
            all_strengths.append(combined_strength)
        
        # 多头输出融合
        multi_head_output = torch.cat(all_head_outputs, dim=-1)  # [B, N, d]
        final_output = self.output_proj(multi_head_output)
        
        # 聚合指针（取第一个头的指针作为主指针）
        main_forward_ptr = all_forward_pointers[0]
        main_backward_ptr = all_backward_pointers[0]
        avg_strength = torch.stack(all_strengths, dim=0).mean(dim=0)
        
        return final_output, main_forward_ptr, main_backward_ptr, avg_strength

class PointerBlock(nn.Module):
    """
    重构的指针块 - 基于双向多头指针机制
    
    核心设计理念：
    1. 双向关系建模：前向和后向指针
    2. 多头机制：不同头关注不同尺度的关系
    3. 多跳支持：可选的指针链传递
    4. 纯关系专注：去除注意力复杂性，专注关系表示
    
    Args:
        d (int): Hidden dimension
        n_heads (int): Number of heads
        n_kv_heads (int): Number of key-value heads (for compatibility)
        max_seq_len (int): Maximum sequence length
        multi_hop (int): Number of hops for pointer chains
    """
    
    def __init__(self, d, n_heads, n_kv_heads=None, use_value_proj=True,
                 use_alibi=False, max_seq_len=4096, addressing_mode='learned',
                 multi_hop=1):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d // n_heads  
        self.max_seq_len = max_seq_len
        
        assert d % n_heads == 0, f"Hidden dim {d} must be divisible by n_heads {n_heads}"
        
        # 核心：双向多头指针机制
        self.bidirectional_pointer = BiDirectionalMultiHeadPointer(d, n_heads, max_seq_len)
        
        # 多跳指针支持
        self.multi_hop = multi_hop
        self.pointer_chain = PointerChain(d, max_hops=multi_hop) if multi_hop > 1 else None
        
        # 兼容性：保留原有的输出投影
        self.o_proj = nn.Linear(d, d, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        init_std = 0.02 / math.sqrt(self.d)
        if hasattr(self.o_proj, 'weight'):
            nn.init.normal_(self.o_proj.weight, mean=0.0, std=init_std)
    
    def forward(self, h, kv_cache=None, prev_idx=None, return_full_scores=False):
        """
        双向多头指针前向传播
        
        Args:
            h (torch.Tensor): Input hidden states [B, N, d]
            kv_cache (Optional): KV cache for inference (简化处理)
            prev_idx (Optional[torch.Tensor]): Previous layer relation targets [B, N] for chaining
            return_full_scores (bool): Whether to return full position scores (兼容性)
            
        Returns:
            Tuple containing:
                - z (torch.Tensor): Output representations [B, N, d]
                - main_pointer (torch.Tensor): Main pointer targets [B, N]
                - relation_strength (torch.Tensor): Relation strength [B, N]
                - full_scores (Optional): Full scores if requested (for compatibility)
        """
        B, N, d = h.shape
        
        # 处理缓存（简化版本）
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
            main_pointer = torch.zeros(B, N, dtype=torch.long, device=h.device)
            relation_strength = torch.zeros(B, N, device=h.device)
            if return_full_scores:
                full_scores = torch.zeros(B, N, N_cache, device=h.device)
                return z, main_pointer, relation_strength, full_scores
            else:
                return z, main_pointer, relation_strength
        
        # 🎯 核心：双向多头指针计算
        pointer_output, forward_ptr, backward_ptr, relation_strength = self.bidirectional_pointer(h, prev_idx)
        
        # 多跳指针链生成（可选）
        if self.multi_hop > 1 and self.pointer_chain is not None:
            forward_chain = self.pointer_chain(h, forward_ptr)
            main_pointer = forward_chain[-1]  # 最后一跳作为主指针
        else:
            main_pointer = forward_ptr  # 前向指针作为主指针
        
        # 🚀 输出投影
        z = self.o_proj(pointer_output)
        
        if return_full_scores:
            # 为兼容性创建全分数矩阵
            full_scores = torch.zeros(B, N, N_cache, device=h.device)
            
            # 在对应的关系目标位置设置强度
            batch_idx = torch.arange(B, device=h.device)[:, None]
            seq_idx = torch.arange(N, device=h.device)[None, :]
            main_ptr_clamped = torch.clamp(main_pointer, 0, N_cache - 1)
            full_scores[batch_idx, seq_idx, main_ptr_clamped] = relation_strength
            
            return z, main_pointer, relation_strength, full_scores
        else:
            return z, main_pointer, relation_strength
