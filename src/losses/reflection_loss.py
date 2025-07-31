"""
Pointer-based Reflection Loss Implementation

基于指针链的真正反思损失，而非简单的序列重复
核心思想：通过指针回顾机制提升模型的推理能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class PointerReflectionLoss(nn.Module):
    """基于指针的反思损失函数"""
    
    def __init__(self, 
                 reflection_weight: float = 0.1,
                 consistency_weight: float = 0.05,
                 selection_weight: float = 0.02):
        super().__init__()
        self.reflection_weight = reflection_weight
        self.consistency_weight = consistency_weight  
        self.selection_weight = selection_weight
        
    def forward(self, 
                logits: torch.Tensor,
                labels: torch.Tensor,
                reflection_outputs: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        计算包含反思损失的总损失
        
        Args:
            logits: 模型输出logits [B, N, vocab_size]
            labels: 标签 [B, N]
            reflection_outputs: 反思层的输出信息
                - layer_history: List[Tensor] 每层的隐状态历史
                - reflection_gates: List[Tensor] 反思门控值
                - reflection_features: List[Tensor] 反思特征
                - reflection_layers: List[int] 反思层索引
        
        Returns:
            损失字典包含总损失和各组件损失
        """
        
        # 1. 标准语言模型损失
        lm_loss = self._compute_language_model_loss(logits, labels)
        total_loss = lm_loss
        
        loss_dict = {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'reflection_loss': torch.tensor(0.0, device=logits.device),
            'consistency_loss': torch.tensor(0.0, device=logits.device),
            'selection_loss': torch.tensor(0.0, device=logits.device)
        }
        
        # 2. 如果有反思输出，计算反思损失
        if reflection_outputs is not None:
            reflection_loss = self._compute_reflection_loss(reflection_outputs)
            consistency_loss = self._compute_consistency_loss(reflection_outputs)
            selection_loss = self._compute_selection_loss(reflection_outputs)
            
            # 加权组合
            total_reflection_loss = (
                self.reflection_weight * reflection_loss +
                self.consistency_weight * consistency_loss +
                self.selection_weight * selection_loss
            )
            
            total_loss = total_loss + total_reflection_loss
            
            loss_dict.update({
                'loss': total_loss,
                'reflection_loss': reflection_loss,
                'consistency_loss': consistency_loss,
                'selection_loss': selection_loss
            })
        
        return loss_dict
    
    def _compute_language_model_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算标准语言模型损失"""
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Cross entropy
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(shift_logits, shift_labels)
    
    def _compute_reflection_loss(self, reflection_outputs: Dict) -> torch.Tensor:
        """
        计算反思损失：鼓励反思层更好地利用历史信息
        
        核心思想：反思后的表征应该包含更丰富的历史信息
        """
        layer_history = reflection_outputs.get('layer_history', [])
        reflection_features = reflection_outputs.get('reflection_features', [])
        
        if not layer_history or not reflection_features:
            return torch.tensor(0.0, device=next(iter(reflection_outputs.values())).device)
        
        total_reflection_loss = 0.0
        num_reflection_layers = len(reflection_features)
        
        for i, (hist_states, refl_feat) in enumerate(zip(layer_history, reflection_features)):
            if len(hist_states) == 0:
                continue
                
            # 计算历史状态的多样性 (我们希望反思能捕获多样的历史信息)
            hist_stack = torch.stack(hist_states, dim=0)  # [num_hist, B, N, d]
            hist_diversity = self._compute_diversity_score(hist_stack)
            
            # 计算反思特征与历史的相关性 (反思应该与历史相关)
            correlation = self._compute_correlation_score(refl_feat, hist_stack)
            
            # 反思损失 = 鼓励高相关性和多样性利用
            reflection_loss = -correlation * hist_diversity
            total_reflection_loss += reflection_loss
        loss = total_reflection_loss / max(num_reflection_layers, 1)
        return (torch.tanh(loss) + 1.0) / 2.0
    
    def _compute_consistency_loss(self, reflection_outputs: Dict) -> torch.Tensor:
        """
        计算一致性损失：反思前后的表征应该保持合理的一致性
        
        避免反思导致信息丢失，同时鼓励有意义的改进
        """
        layer_history = reflection_outputs.get('layer_history', [])
        reflection_features = reflection_outputs.get('reflection_features', [])
        
        if not layer_history or not reflection_features:
            return torch.tensor(0.0, device=next(iter(reflection_outputs.values())).device)
        
        total_consistency_loss = 0.0
        num_pairs = 0
        
        for hist_states, refl_feat in zip(layer_history, reflection_features):
            if len(hist_states) == 0:
                continue
                
            # 使用最近的历史状态作为反思前的表征
            recent_state = hist_states[-1]  # [B, N, d]
            
            # 计算余弦相似度 (保持一定一致性)
            cos_sim = F.cosine_similarity(
                recent_state.view(-1, recent_state.size(-1)),
                refl_feat.view(-1, refl_feat.size(-1)),
                dim=-1
            ).mean()
            
            # 一致性损失：鼓励适度的相似性 (0.7-0.9为理想范围)
            target_similarity = 0.8
            consistency_loss = (cos_sim - target_similarity) ** 2
            
            total_consistency_loss += consistency_loss
            num_pairs += 1
        
        return total_consistency_loss / max(num_pairs, 1)
    
    def _compute_selection_loss(self, reflection_outputs: Dict) -> torch.Tensor:
        """
        计算选择损失：鼓励模型有选择性地使用反思门控
        
        避免反思门控总是开启或关闭，鼓励智能选择
        """
        reflection_gates = reflection_outputs.get('reflection_gates', [])
        
        if not reflection_gates:
            return torch.tensor(0.0, device=next(iter(reflection_outputs.values())).device)
        
        total_selection_loss = 0.0
        
        for gate_values in reflection_gates:
            # gate_values: [B, N, d] 或 [B, N, 1]
            if gate_values.dim() > 2:
                gate_values = gate_values.mean(dim=-1)  # [B, N]
            
            # 计算门控值的方差 (鼓励多样化使用)
            gate_variance = torch.var(gate_values, dim=-1).mean()  # 跨序列位置的方差
            
            # 计算门控值的均值 (避免全开或全关)
            gate_mean = torch.mean(gate_values)
            
            # 选择损失：鼓励适中的均值(0.3-0.7)和高方差(多样性)
            mean_penalty = torch.min(
                (gate_mean - 0.3) ** 2,
                (gate_mean - 0.7) ** 2
            )
            variance_reward = -gate_variance  # 负号表示奖励高方差
            
            selection_loss = mean_penalty + 0.1 * variance_reward
            total_selection_loss += selection_loss
        
        return total_selection_loss / len(reflection_gates)
    
    def _compute_diversity_score(self, hist_stack: torch.Tensor) -> torch.Tensor:
        """计算历史状态的多样性分数"""
        # hist_stack: [num_hist, B, N, d]
        if hist_stack.size(0) < 2:
            return torch.tensor(1.0, device=hist_stack.device)
        
        # 计算不同历史层之间的余弦距离
        flat_hist = hist_stack.view(hist_stack.size(0), -1)  # [num_hist, B*N*d]
        
        # 计算两两相似度矩阵
        norm_hist = F.normalize(flat_hist, dim=-1)
        similarity_matrix = torch.mm(norm_hist, norm_hist.t())
        
        # 多样性 = 1 - 平均相似度 (除了对角线)
        mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=hist_stack.device)
        avg_similarity = similarity_matrix[mask].mean()
        diversity = 1.0 - avg_similarity
        
        return torch.clamp(diversity, 0.0, 1.0)
    
    def _compute_correlation_score(self, refl_feat: torch.Tensor, hist_stack: torch.Tensor) -> torch.Tensor:
        """计算反思特征与历史的相关性分数"""
        # refl_feat: [B, N, d]
        # hist_stack: [num_hist, B, N, d]
        
        # 展平用于计算相关性
        flat_refl = refl_feat.view(-1, refl_feat.size(-1))  # [B*N, d]
        flat_hist = hist_stack.view(hist_stack.size(0), -1)  # [num_hist, B*N*d]
        
        # 计算反思特征与每个历史层的相关性
        correlations = []
        for i in range(hist_stack.size(0)):
            hist_layer = hist_stack[i].view(-1, hist_stack.size(-1))  # [B*N, d]
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(flat_refl, hist_layer, dim=-1).mean()
            correlations.append(cos_sim)
        
        # 返回最大相关性 (反思应该与某些历史层高度相关)
        return torch.stack(correlations).max()


def create_reflection_loss(reflection_weight: float = 0.1,
                          consistency_weight: float = 0.05,
                          selection_weight: float = 0.02) -> PointerReflectionLoss:
    """创建反思损失函数的工厂函数"""
    return PointerReflectionLoss(
        reflection_weight=reflection_weight,
        consistency_weight=consistency_weight,
        selection_weight=selection_weight
    )