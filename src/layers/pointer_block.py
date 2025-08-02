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


class PointerBlock(nn.Module):
    """
    纯关系建模块 - 支持多跳关系链 (A→B→C→...)
    
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
    
    def __init__(self, d, n_heads, n_kv_heads=None, use_value_proj=True,
                 use_alibi=False, max_seq_len=4096, addressing_mode='learned',
                 multi_hop=1, dynamic_threshold=0.3, max_branches=3):  # 动态分叉参数
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
        
        # 多跳指针支持
        self.multi_hop = multi_hop
        self.pointer_chain = PointerChain(d, max_hops=multi_hop) if multi_hop > 1 else None
        
        # 可学习的动态分叉参数
        self.dynamic_threshold = nn.Parameter(torch.tensor(dynamic_threshold))
        self.max_branches = max_branches
        # 动态分叉学习网络
        self.branch_learner = nn.Sequential(
            nn.Linear(d, d//2),
            nn.GELU(),
            nn.Linear(d//2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        init_std = 0.02 / math.sqrt(self.d)
        for module in [self.value_proj, self.o_proj]:
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
    
    def _compute_pure_relations(self, h, prev_idx=None):
        """动态分叉关系建模
        
        Args:
            h (torch.Tensor): Hidden states [B, N, d]
            prev_idx (Optional[torch.Tensor]): Previous layer's relation chain [B, N]
            
        Returns:
            List[torch.Tensor]: 动态生成的分叉指针列表 [B, N]
        """
        B, N, d = h.shape
        device = h.device
        
        # 计算基础关系强度
        base_logits = self.relation_encoder(h)  # [B, N, 1]
        base_strength = torch.sigmoid(base_logits)  # [B, N, 1]
        
        # 动态分叉决策
        branch_mask = (base_strength > self.dynamic_threshold).float()  # [B, N, 1]
        num_branches = torch.clamp(
            (base_strength / self.dynamic_threshold).round().long(),
            1, self.max_branches
        )  # [B, N, 1]
        
        # 生成多分支指针
        all_pointers = []
        for b in range(self.max_branches):
            # 每个分支有轻微不同的关系计算
            branch_logits = self.relation_encoder(h + 0.1*b)  # [B, N, 1]
            branch_logits = branch_logits.squeeze(-1)  # [B, N]
            branch_targets = torch.sigmoid(branch_logits) * (N - 1)
            branch_targets = branch_targets.round().long().view(B, N)  # 确保形状为[B, N]
            
            # 只保留有效的分支
            active = (b < num_branches).squeeze(-1)  # [B, N]
            # 确保zeros_like与branch_targets维度一致
            zeros = torch.zeros_like(branch_targets)
            branch_targets = torch.where(
                active, 
                branch_targets,
                zeros)  # 无效分支指向0
                
            all_pointers.append(branch_targets)
        
        # 主指针总是第一个分支
        main_ptr = all_pointers[0]
        
        # 关系链继承
        if prev_idx is not None:
            chain_threshold = N // 2
            should_chain = torch.arange(N, device=device) >= chain_threshold
            should_chain = should_chain.unsqueeze(0).expand(B, N)
            prev_idx_clamped = torch.clamp(prev_idx, 0, N - 1)
            main_ptr = torch.where(should_chain, prev_idx_clamped, main_ptr)
        
        # 更新第一个分支
        all_pointers[0] = main_ptr
        
        return all_pointers  # List[[B, N], ...]
    
    def _pure_relation_aggregation(self, h, relation_targets):
        """
        🔥 纯关系信息聚合：支持动态分叉
        
        Args:
            h (torch.Tensor): Source hidden states [B, N, d]
            relation_targets (torch.Tensor | List[torch.Tensor]): 
                 单跳[B, N]或多跳指针链List[[B, N], ...]
            
        Returns:
            torch.Tensor: Relation-aggregated features [B, N, d]
        """
        # 处理多跳情况
        if isinstance(relation_targets, list) and len(relation_targets) > 0 and isinstance(relation_targets[0], list):
            # 多跳模式
            all_relation_feats = []
            for ptr_chain in relation_targets:
                chain_feats = []
                for ptr in ptr_chain:
                    target_feat = gather_by_pointer(h, ptr)
                    source_feat = self.value_proj(h)
                    target_feat = self.value_proj(target_feat)
                    chain_feats.append(torch.cat([source_feat, target_feat], dim=-1))
                all_relation_feats.append(torch.mean(torch.stack(chain_feats), dim=0))
            relation_input = torch.mean(torch.stack(all_relation_feats), dim=0)
        elif isinstance(relation_targets, list):  # 单指针多跳模式
            # 最后一跳作为主关系
            main_ptr = relation_targets[-1]
            # 聚合多跳信息
            multi_hop_feats = []
            for ptr in relation_targets:
                target_feat = gather_by_pointer(h, ptr)
                source_feat = self.value_proj(h)
                target_feat = self.value_proj(target_feat)
                relation_feat = torch.cat([source_feat, target_feat], dim=-1)
                multi_hop_feats.append(relation_feat)
            # 平均多跳特征
            relation_input = torch.mean(torch.stack(multi_hop_feats), dim=0)
        else:  # 单跳模式
            target_feat = gather_by_pointer(h, relation_targets)
            source_feat = self.value_proj(h)
            target_feat = self.value_proj(target_feat)
            relation_input = torch.cat([source_feat, target_feat], dim=-1)
        
        return self.relation_transform(relation_input)
    
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
        
        # 🎯 步骤1：学习纯关系
        first_hop = self._compute_pure_relations(h, prev_idx)  # [B, N] 或 List[[B, N],...]
        
        # 多跳指针链生成
        if self.multi_hop > 1 and self.pointer_chain is not None:
            if isinstance(first_hop, list):  # 多指针模式
                relation_targets = [self.pointer_chain(h, ptr) for ptr in first_hop]
            else:  # 单指针模式
                relation_targets = self.pointer_chain(h, first_hop)
        else:
            relation_targets = first_hop
        
        # 🔥 步骤2：关系信息聚合 (自动处理单跳/多跳)
        relation_output = self._pure_relation_aggregation(h, relation_targets)  # [B, N, d]
        
        # 统一返回格式
        if isinstance(relation_targets, list) and isinstance(relation_targets[0], list):
            # 多指针+多跳模式: 返回第一个指针链作为主指针
            main_ptr = relation_targets[0][-1]  # 取第一个指针链的最后一跳
        elif isinstance(relation_targets, list):
            # 多指针单跳模式: 返回第一个指针作为主指针
            main_ptr = relation_targets[0]
        else:
            # 单指针模式
            main_ptr = relation_targets
        
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
            
            return z, main_ptr, relation_strength, full_scores
        else:
            return z, main_ptr, relation_strength
