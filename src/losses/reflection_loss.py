"""
Pointer-based Reflection Loss Implementation

True reflection loss based on pointer chains, not simple sequence repetition
Core idea: Enhance model reasoning capabilities through pointer review mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class PointerReflectionLoss(nn.Module):
    """Pointer-based reflection loss function"""
    
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
        Compute total loss including reflection loss
        
        Args:
            logits: Model output logits [B, N, vocab_size]
            labels: Labels [B, N]
            reflection_outputs: Reflection layer output information
                - layer_history: List[Tensor] Hidden state history for each layer
                - reflection_gates: List[Tensor] Reflection gate values
                - reflection_features: List[Tensor] Reflection features
                - reflection_layers: List[int] Reflection layer indices
        
        Returns:
            Loss dictionary containing total loss and component losses
        """
        
        # 1. Standard language model loss
        lm_loss = self._compute_language_model_loss(logits, labels)
        total_loss = lm_loss
        
        loss_dict = {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'reflection_loss': torch.tensor(0.0, device=logits.device),
            'consistency_loss': torch.tensor(0.0, device=logits.device),
            'selection_loss': torch.tensor(0.0, device=logits.device)
        }
        
        # 2. If reflection outputs exist, compute reflection loss
        if reflection_outputs is not None:
            reflection_loss = self._compute_reflection_loss(reflection_outputs)
            consistency_loss = self._compute_consistency_loss(reflection_outputs)
            selection_loss = self._compute_selection_loss(reflection_outputs)
            
            # Weighted combination
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
        """Compute standard language model loss"""
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
        Compute reflection loss: Encourage reflection layers to better utilize historical information
        
        Core idea: Post-reflection representations should contain richer historical information
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
                
            # Compute diversity of historical states (we want reflection to capture diverse historical info)
            hist_stack = torch.stack(hist_states, dim=0)  # [num_hist, B, N, d]
            hist_diversity = self._compute_diversity_score(hist_stack)
            
            # Compute correlation between reflection features and history (reflection should be history-related)
            correlation = self._compute_correlation_score(refl_feat, hist_stack)
            
            # Reflection loss = encourage high correlation and diversity utilization
            reflection_loss = -correlation * hist_diversity
            total_reflection_loss += reflection_loss
        loss = total_reflection_loss / max(num_reflection_layers, 1)
        return (torch.tanh(loss) + 1.0) / 2.0
    
    def _compute_consistency_loss(self, reflection_outputs: Dict) -> torch.Tensor:
        """
        Compute consistency loss: Pre and post-reflection representations should maintain reasonable consistency
        
        Avoid information loss due to reflection while encouraging meaningful improvements
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
                
            # Use recent historical state as pre-reflection representation
            recent_state = hist_states[-1]  # [B, N, d]
            
            # Compute cosine similarity (maintain certain consistency)
            cos_sim = F.cosine_similarity(
                recent_state.view(-1, recent_state.size(-1)),
                refl_feat.view(-1, refl_feat.size(-1)),
                dim=-1
            ).mean()
            
            # Consistency loss: encourage moderate similarity (0.7-0.9 is ideal range)
            target_similarity = 0.8
            consistency_loss = (cos_sim - target_similarity) ** 2
            
            total_consistency_loss += consistency_loss
            num_pairs += 1
        
        return total_consistency_loss / max(num_pairs, 1)
    
    def _compute_selection_loss(self, reflection_outputs: Dict) -> torch.Tensor:
        """
        Compute selection loss: Encourage model to selectively use reflection gating
        
        Avoid reflection gates being always on or off, encourage intelligent selection
        """
        reflection_gates = reflection_outputs.get('reflection_gates', [])
        
        if not reflection_gates:
            return torch.tensor(0.0, device=next(iter(reflection_outputs.values())).device)
        
        total_selection_loss = 0.0
        
        for gate_values in reflection_gates:
            # gate_values: [B, N, d] or [B, N, 1]
            if gate_values.dim() > 2:
                gate_values = gate_values.mean(dim=-1)  # [B, N]
            
            # Compute gate value variance (encourage diversified usage)
            gate_variance = torch.var(gate_values, dim=-1).mean()  # Variance across sequence positions
            
            # Compute gate value mean (avoid all-on or all-off)
            gate_mean = torch.mean(gate_values)
            
            # Selection loss: encourage moderate mean (0.3-0.7) and high variance (diversity)
            mean_penalty = torch.min(
                (gate_mean - 0.3) ** 2,
                (gate_mean - 0.7) ** 2
            )
            variance_reward = -gate_variance  # Negative sign means rewarding high variance
            
            selection_loss = mean_penalty + 0.1 * variance_reward
            total_selection_loss += selection_loss
        
        return total_selection_loss / len(reflection_gates)
    
    def _compute_diversity_score(self, hist_stack: torch.Tensor) -> torch.Tensor:
        """Compute diversity score of historical states"""
        # hist_stack: [num_hist, B, N, d]
        if hist_stack.size(0) < 2:
            return torch.tensor(1.0, device=hist_stack.device)
        
        # Compute cosine distance between different historical layers
        flat_hist = hist_stack.view(hist_stack.size(0), -1)  # [num_hist, B*N*d]
        
        # Compute pairwise similarity matrix
        norm_hist = F.normalize(flat_hist, dim=-1)
        similarity_matrix = torch.mm(norm_hist, norm_hist.t())
        
        # Diversity = 1 - average similarity (excluding diagonal)
        mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=hist_stack.device)
        avg_similarity = similarity_matrix[mask].mean()
        diversity = 1.0 - avg_similarity
        
        return torch.clamp(diversity, 0.0, 1.0)
    
    def _compute_correlation_score(self, refl_feat: torch.Tensor, hist_stack: torch.Tensor) -> torch.Tensor:
        """Compute correlation score between reflection features and history"""
        # refl_feat: [B, N, d]
        # hist_stack: [num_hist, B, N, d]
        
        # Flatten for correlation computation
        flat_refl = refl_feat.view(-1, refl_feat.size(-1))  # [B*N, d]
        flat_hist = hist_stack.view(hist_stack.size(0), -1)  # [num_hist, B*N*d]
        
        # Compute correlation between reflection features and each historical layer
        correlations = []
        for i in range(hist_stack.size(0)):
            hist_layer = hist_stack[i].view(-1, hist_stack.size(-1))  # [B*N, d]
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(flat_refl, hist_layer, dim=-1).mean()
            correlations.append(cos_sim)
        
        # Return maximum correlation (reflection should be highly correlated with some historical layers)
        return torch.stack(correlations).max()


def create_reflection_loss(reflection_weight: float = 0.1,
                          consistency_weight: float = 0.05,
                          selection_weight: float = 0.02) -> PointerReflectionLoss:
    """Factory function to create reflection loss function"""
    return PointerReflectionLoss(
        reflection_weight=reflection_weight,
        consistency_weight=consistency_weight,
        selection_weight=selection_weight
    )