import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List

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


class PointerChainReflection(nn.Module):
    """
    基于指针链的反思机制 - 轻量级、可解释的指针历史分析
    
    核心理念：
    1. 只存储指针索引，内存开销从 O(L*N*d) 降到 O(L*N)
    2. 分析指针链模式：自循环、长跳、收敛等
    3. 基于指针关系历史生成当前层的指针调整建议
    4. 完全可解释：每个反思决策都可追溯到具体指针路径
    
    Args:
        d (int): Hidden dimension (用于特征编码)
        max_history_layers (int): 最大历史层数
        pattern_analysis_dim (int): 指针模式分析的特征维度
    """
    
    def __init__(self, d, max_history_layers=8, pattern_analysis_dim=64):
        super().__init__()
        self.d = d
        self.max_history_layers = max_history_layers
        self.pattern_dim = pattern_analysis_dim
        
        # 🔍 指针模式识别网络
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(max_history_layers, pattern_analysis_dim),
            nn.GELU(),
            nn.Linear(pattern_analysis_dim, pattern_analysis_dim // 2),
            nn.GELU(),
            nn.Linear(pattern_analysis_dim // 2, 4)  # 4种基本模式：自循环、短跳、长跳、随机
        )
        
        # 🎯 基于模式的指针调整生成器
        self.pointer_adjustment_generator = nn.Sequential(
            nn.Linear(4 + d, d // 2),  # 模式特征 + 当前隐状态
            nn.GELU(),  
            nn.Linear(d // 2, 1)  # 生成指针调整建议
        )
        
        # 🧠 可学习的模式权重
        self.self_loop_weight = nn.Parameter(torch.tensor(0.1))      # 自循环模式权重
        self.short_jump_weight = nn.Parameter(torch.tensor(0.3))     # 短跳模式权重  
        self.long_jump_weight = nn.Parameter(torch.tensor(0.4))      # 长跳模式权重
        self.convergence_weight = nn.Parameter(torch.tensor(0.2))    # 收敛模式权重
        
        # 🔄 反思强度控制
        self.reflection_intensity = nn.Parameter(torch.tensor(0.2))
        
        # 🎯 NEW: 可学习的距离阈值 (替代硬编码)
        self.short_jump_threshold = nn.Parameter(torch.tensor(2.0))   # 短跳距离阈值
        self.long_jump_ratio = nn.Parameter(torch.tensor(0.25))       # 长跳距离比例 (相对于序列长度)
        
        # 🧮 可学习的模式计算权重
        self.pattern_consistency_weight = nn.Parameter(torch.tensor(1.0))    # 一致性权重
        self.pattern_diversity_weight = nn.Parameter(torch.tensor(0.5))      # 多样性权重
        
        print(f"PointerChainReflection initialized: max_history={max_history_layers}, pattern_dim={pattern_analysis_dim}, learnable_thresholds=True")
    
    def _analyze_pointer_patterns_vectorized(self, pointer_history: List[torch.Tensor]) -> torch.Tensor:
        """
        向量化分析指针链中的模式 - 一次性处理所有位置
        
        Args:
            pointer_history: 历史指针列表 [Tensor[B, N], ...]
            
        Returns:
            pattern_features: 模式特征 [B, N, 4] (自循环、短跳、长跳、收敛)
        """
        if not pointer_history:
            # 如果没有历史，返回零模式
            return torch.zeros(1, 1, 4, device=torch.device('cpu'))
        
        B, N = pointer_history[0].shape
        device = pointer_history[0].device
        
        # 提取历史指针矩阵 [B, N, L]
        history_matrix = []
        for ptr_tensor in pointer_history[-self.max_history_layers:]:
            if ptr_tensor.size(1) == N:
                history_matrix.append(ptr_tensor)
            else:
                # 处理尺寸不匹配的情况
                padded = torch.zeros(B, N, device=device, dtype=torch.long)
                min_len = min(N, ptr_tensor.size(1))
                padded[:, :min_len] = ptr_tensor[:, :min_len]
                history_matrix.append(padded)
        
        if not history_matrix:
            return torch.zeros(B, N, 4, device=device)
        
        history_matrix = torch.stack(history_matrix, dim=2)  # [B, N, L]
        B, N, L = history_matrix.shape
        
        # 🚀 向量化模式分析
        patterns = torch.zeros(B, N, 4, device=device)
        
        # 创建位置索引矩阵 [B, N]
        position_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # [B, N]
        
        # 1. 自循环模式：向量化计算所有位置的自循环频率
        self_loops = (history_matrix == position_indices.unsqueeze(2)).float().mean(dim=2)  # [B, N]
        patterns[:, :, 0] = self_loops
        
        # 2. 短跳模式：向量化距离计算
        short_threshold = torch.relu(self.short_jump_threshold)
        distances = torch.abs(history_matrix.float() - position_indices.unsqueeze(2).float())  # [B, N, L]
        is_short_jump = (distances <= short_threshold) & (history_matrix != position_indices.unsqueeze(2))
        short_jumps = is_short_jump.float().mean(dim=2)  # [B, N]
        patterns[:, :, 1] = short_jumps
        
        # 3. 长跳模式：向量化相对阈值计算
        long_threshold = N * torch.sigmoid(self.long_jump_ratio)
        is_long_jump = distances > long_threshold
        long_jumps = is_long_jump.float().mean(dim=2)  # [B, N]
        patterns[:, :, 2] = long_jumps
        
        # 4. 收敛模式：向量化方差计算
        if L > 1:
            # 计算每个位置历史的方差 [B, N]
            variance = torch.var(history_matrix.float(), dim=2)  # [B, N]
            consistency_factor = torch.sigmoid(self.pattern_consistency_weight)
            diversity_factor = torch.sigmoid(self.pattern_diversity_weight)
            convergence = consistency_factor / (1.0 + diversity_factor * variance)
            patterns[:, :, 3] = convergence
        else:
            patterns[:, :, 3] = 0.5  # 中性值
        
        return patterns
    
    def _compute_pointer_stability(self, pointer_history: List[torch.Tensor]) -> torch.Tensor:
        """
        计算指针链的稳定性 - 衡量指针选择的一致性 (使用可学习参数)
        
        Args:
            pointer_history: 历史指针列表
            
        Returns:
            stability: 稳定性分数 [B, N]
        """
        if len(pointer_history) < 2:
            B, N = pointer_history[0].shape if pointer_history else (1, 1)
            device = pointer_history[0].device if pointer_history else torch.device('cpu')
            return torch.ones(B, N, device=device) * 0.5  # 中性稳定性
        
        # 可学习的稳定性窗口大小
        window_size = max(2, min(len(pointer_history), int(torch.relu(self.pattern_consistency_weight) * 3) + 2))
        recent_history = pointer_history[-window_size:]  # 取最近几层
        consistency_scores = []
        
        for i in range(len(recent_history) - 1):
            # 计算相邻层指针的相似度
            ptr1, ptr2 = recent_history[i], recent_history[i + 1]
            consistency = (ptr1 == ptr2).float()  # [B, N]
            consistency_scores.append(consistency)
        
        if consistency_scores:
            # 使用可学习权重计算稳定性
            weights = torch.softmax(torch.tensor([torch.sigmoid(self.pattern_diversity_weight)] * len(consistency_scores)), dim=0)
            stability = sum(w * score for w, score in zip(weights, consistency_scores))  # [B, N]
        else:
            B, N = recent_history[0].shape
            device = recent_history[0].device
            stability = torch.ones(B, N, device=device) * 0.5
        
        return stability
    
    def forward(self, h: torch.Tensor, pointer_history: List[torch.Tensor]) -> torch.Tensor:
        """
        基于指针链历史生成反思增强的特征 - 完全向量化版本
        
        Args:
            h: 当前隐状态 [B, N, d]
            pointer_history: 指针历史 [Tensor[B, N], ...]
            
        Returns:
            reflected_h: 反思增强的隐状态 [B, N, d]
        """
        if not pointer_history:
            return h
        
        B, N, d = h.shape
        device = h.device
        
        # 🚀 向量化模式分析 - 一次性处理所有位置
        all_patterns = self._analyze_pointer_patterns_vectorized(pointer_history)  # [B, N, 4]
        
        # 🎯 向量化指针调整生成
        # 将隐状态和模式特征组合 [B, N, d+4]
        combined_input = torch.cat([all_patterns, h], dim=-1)  # [B, N, 4+d]
        
        # 批量通过调整生成器
        adjustments = self.pointer_adjustment_generator(combined_input)  # [B, N, 1]
        
        # 🧠 向量化模式权重计算
        pattern_weights = torch.stack([
            self.self_loop_weight,
            self.short_jump_weight, 
            self.long_jump_weight,
            self.convergence_weight
        ], dim=0)  # [4]
        
        # 计算加权模式分数 [B, N, 1]
        weighted_patterns = all_patterns * pattern_weights.view(1, 1, 4)  # [B, N, 4]
        pattern_influence = weighted_patterns.sum(dim=-1, keepdim=True)  # [B, N, 1]
        
        # 🔄 动态反思强度
        dynamic_intensity = torch.sigmoid(self.reflection_intensity) * pattern_influence
        
        # 最终反思增强 - 向量化计算
        h_norm = h.norm(dim=-1, keepdim=True)  # [B, N, 1]
        reflection_delta = dynamic_intensity * adjustments * h_norm
        reflected_h = h + reflection_delta
        
        return reflected_h


class PointerLayer(nn.Module):
    """DeepSeek-style Pointer Layer with Pointer-Chain Reflection Mechanism.
    
    Architecture: RMSNorm → PointerBlock → Residual → RMSNorm → SwiGLU → Residual (Pre-norm)
    Key features: 
    - Passes prev_index to form pointer chains across layers
    - NEW: Pointer-chain based reflection for efficient and interpretable reasoning
    - Lightweight O(L*N) reflection instead of O(L*N*d) 
    
    Args:
        d (int): Hidden dimension
        n_heads (int): Number of attention heads
        layer_idx (int): Current layer index
        n_kv_heads (int): Number of key-value heads
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
        
        # 🧠 NEW: 基于指针链的反思机制
        self.reflection_config = reflection_config or {}
        max_history = self.reflection_config.get('max_history_layers', 8)
        self.pointer_chain_reflection = PointerChainReflection(
            d=d,
            max_history_layers=max_history,
            pattern_analysis_dim=64
        )
        
        # DeepSeek-style RMSNorm (Pre-norm architecture)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        
        # Pointer block
        self.pointer_block = PointerBlock(
            d=d,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_value_proj=use_value_proj,
            use_alibi=use_alibi,
            max_seq_len=max_seq_len,
        )
        
        # Learnable gate for pointer output
        self.gate = nn.Parameter(torch.ones(d))
        
        # Llama-style SwiGLU FFN
        self.ffn = LlamaMLP(
            hidden_size=d,
            intermediate_size=d_ff or int(8 * d / 3),
            hidden_act="silu"
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"PointerLayer {layer_idx} initialized: d={d}, n_heads={n_heads}, "
              f"pointer_chain_reflection=True, max_history={max_history}")
    
    def forward(self, h, kv_cache=None, prev_idx=None, layer_history=None, pointer_history=None, return_full_scores=False):
        """DeepSeek-style forward pass with pointer-chain reflection.
        
        Args:
            h (torch.Tensor): Input hidden states [B, N, d]
            kv_cache (Optional): KV cache for inference
            prev_idx (Optional[torch.Tensor]): Previous layer's pointer indices for chaining
            layer_history (Optional[List[torch.Tensor]]): DEPRECATED - not used in new reflection
            pointer_history (Optional[List[torch.Tensor]]): Pointer indices history for reflection
            return_full_scores (bool): Whether to return full position scores
            
        Returns:
            Tuple containing:
                - h (torch.Tensor): Output hidden states [B, N, d]
                - idx (torch.Tensor): Current layer's pointer indices [B, N] 
                - p (torch.Tensor): Pointer probabilities [B, N]
                - full_scores (Optional[torch.Tensor]): Full position scores if requested
        """
        # 🧠 NEW: 基于指针链的轻量级反思
        if pointer_history:
            h = self.pointer_chain_reflection(h, pointer_history)
        
        # --- Pointer part (Pre-norm) ---
        residual = h
        
        # Pre-norm: normalize then compute
        h_norm = self.norm1(h)
        
        # Pointer computation
        pointer_result = self.pointer_block(h_norm, kv_cache, prev_idx=prev_idx, return_full_scores=return_full_scores)
        
        # Always ensure we have the right number of values
        if return_full_scores:
            if len(pointer_result) == 4:
                z, idx, p, full_scores = pointer_result
            else:
                z, idx, p = pointer_result
                # Create dummy full_scores for compatibility
                B, N = h.shape[:2]
                N_cache = kv_cache.max_seq_len if (kv_cache and hasattr(kv_cache, 'max_seq_len')) else self.pointer_block.max_seq_len
                full_scores = torch.zeros(B, N, N_cache, device=h.device)
        else:
            if len(pointer_result) == 4:
                z, idx, p, _ = pointer_result  # Ignore full_scores
            else:
                z, idx, p = pointer_result
        
        # Apply learnable gate and residual connection
        z = z * self.gate
        h = residual + self.dropout(z)
        
        # --- FFN part (Pre-norm) ---
        residual = h
        h = self.norm2(h)
        h = self.ffn(h)
        h = residual + self.dropout(h)
        
        if return_full_scores:
            return h, idx, p, full_scores
        else:
            return h, idx, p
