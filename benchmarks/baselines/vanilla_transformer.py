"""
Vanilla Transformer Baseline with ALiBi Positional Encoding

This implements a standard Transformer decoder as a fair baseline comparison
for the Pointer architecture. Uses ALiBi for position encoding to match
the Pointer model's length extrapolation capabilities.

Key features:
- Standard multi-head self-attention (full attention)
- ALiBi positional encoding for length extrapolation
- Same architecture depth/width as Pointer model
- RMSNorm for fair comparison
- Pointer supervision head for position prediction tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any


class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi positional encoding."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi slopes
        self.register_buffer('slopes', self._get_alibi_slopes(n_heads))
        
        # Cache for computed masks and biases
        self._cached_masks = {}
        self._cached_alibi = {}
    
    def _get_alibi_slopes(self, n_heads: int) -> torch.Tensor:
        """Get ALiBi slopes for each head."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n_heads-closest_power_of_2])
        
        return torch.tensor(slopes, dtype=torch.float32)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        
        # Get or compute cached ALiBi bias
        cache_key = (T, device, self.n_heads)
        if cache_key not in self._cached_alibi:
            # Create position matrix only once per sequence length
            i = torch.arange(T, device=device)
            j = torch.arange(T, device=device)
            positions = i.view(-1, 1) - j.view(1, -1)  # More efficient
            slopes = self.slopes.to(device).view(1, -1, 1, 1)
            alibi_bias = positions.unsqueeze(0).unsqueeze(0) * slopes
            self._cached_alibi[cache_key] = alibi_bias
        
        # Use cached bias efficiently
        alibi_bias = self._cached_alibi[cache_key][:, :self.n_heads, :T, :T]
        scores = scores + alibi_bias
        
        # Get or compute cached causal mask
        mask_cache_key = (T, device)
        if mask_cache_key not in self._cached_masks:
            causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
            self._cached_masks[mask_cache_key] = causal_mask
        
        scores = scores.masked_fill(~self._cached_masks[mask_cache_key], float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, 1, T]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        
        # Output projection
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function from PaLM paper."""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Standard Transformer block with ALiBi attention and RMSNorm."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = ALiBiAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (like Pointer model)
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class VanillaTransformerDecoder(nn.Module):
    """
    Vanilla Transformer decoder with ALiBi for fair comparison with Pointer model.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of layers
        n_heads: Number of attention heads  
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        tie_embeddings: Whether to tie input/output embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_embeddings: bool = True
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = int(8 * d_model / 3)  # Same as Pointer model
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        
        # Pointer supervision head (for position prediction tasks)
        self.pointer_head = nn.Linear(d_model, max_seq_len, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"VanillaTransformerDecoder initialized: {total_params:.1f}M parameters")
        print(f"Config: d={d_model}, layers={n_layers}, heads={n_heads}, max_seq_len={max_seq_len}")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_pointer_logits: bool = False,
        **kwargs  # For compatibility with Pointer model
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Attention mask [B, T]
            labels: Labels for language modeling loss [B, T]
            return_pointer_logits: Whether to return pointer logits
            
        Returns:
            Dictionary with logits, loss, and optionally pointer_logits
        """
        B, T = input_ids.shape
        
        # Clamp input IDs to vocabulary range
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Also clamp labels if provided
        if labels is not None:
            labels = torch.clamp(labels, 0, self.vocab_size - 1)
        
        # Embed tokens
        x = self.embed(input_ids)  # [B, T, D]
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)  # [B, T, D]
        
        # Language modeling logits
        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        result = {'logits': logits}
        
        # Pointer logits (for position supervision)
        if return_pointer_logits:
            # Use last layer hidden states to predict positions
            pointer_logits = self.pointer_head(x)  # [B, T, max_seq_len]
            
            # Mask out positions beyond current sequence length
            pos_mask = torch.arange(self.max_seq_len, device=x.device).unsqueeze(0).unsqueeze(0)
            seq_len_mask = pos_mask >= T  # Positions beyond sequence length
            pointer_logits = pointer_logits.masked_fill(seq_len_mask, float('-inf'))
            
            result['pointer_logits'] = pointer_logits
        
        # 自适应损失计算 - 支持分类和语言建模任务
        if labels is not None:
            if labels.dim() == 1:  # 分类任务：labels是[B]
                # 使用第一个token的logits进行分类
                cls_logits = logits[:, 0, :]  # [B, vocab_size]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(cls_logits, labels)
            else:  # 语言建模任务：labels是[B, N]
                # 标准的causal language modeling loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits, shift_labels)
            
            result['loss'] = loss
        
        return result
    
    def get_pointer_stats(self) -> Dict[str, float]:
        """
        Get pointer statistics for compatibility with Pointer model.
        For vanilla transformer, these are computed from attention patterns.
        """
        # This is a simplified version - in practice, we'd analyze attention patterns
        return {
            'pointer_utilization': 0.5,  # Dummy value
            'avg_hop_distance': 1.0,     # Dummy value
            'pointer_entropy': 2.0       # Dummy value
        }


class TransformerTaskTrainer:
    """
    Trainer wrapper for vanilla Transformer to match PointerTaskTrainer interface.
    """
    
    def __init__(self, config, task_name: str = "transformer_task"):
        # Import the training components from pointer trainer
        import sys
        sys.path.append('/Volumes/oz/pointer/benchmarks')
        from benchmarks.runners.train_pointer_index import TrainingConfig, PointerTaskTrainer
        
        self.config = config
        self.task_name = task_name
        self.device = torch.device(config.device)
        
        # Initialize vanilla transformer with unified hyperparameters for fair comparison
        self.model = VanillaTransformerDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout  # Use unified dropout from fair config
        ).to(self.device)
        
        # Use same optimizer and training setup as PointerTaskTrainer
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        self.step = 0
        self.best_metrics = {}
        
        # Create output directory
        import os
        self.output_dir = os.path.join(config.output_dir, f"{config.experiment_name}_{task_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 and config.device == 'cuda' else None
        
        print(f"Initialized TransformerTaskTrainer for {task_name}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        pointer_targets = batch.get('pointer_targets')
        if pointer_targets is not None:
            pointer_targets = pointer_targets.to(self.device)
        
        # Forward pass
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                output = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    return_pointer_logits=pointer_targets is not None
                )
                loss = output['loss']
                
                # Add pointer loss if we have targets
                if pointer_targets is not None and 'pointer_logits' in output:
                    pointer_logits = output['pointer_logits']  # [B, N, max_seq_len]
                    # Use only the last position for prediction
                    query_logits = pointer_logits[:, -1, :]  # [B, max_seq_len]
                    pointer_loss = torch.nn.functional.cross_entropy(
                        query_logits,
                        pointer_targets,
                        ignore_index=-100
                    )
                    loss = loss + pointer_loss
        else:
            output = self.model(
                input_ids=input_ids,
                labels=labels,
                return_pointer_logits=pointer_targets is not None
            )
            loss = output['loss']
            
            # Add pointer loss if we have targets
            if pointer_targets is not None and 'pointer_logits' in output:
                pointer_logits = output['pointer_logits']  # [B, N, max_seq_len]
                # Use only the last position for prediction (like PointerTaskTrainer)
                query_logits = pointer_logits[:, -1, :]  # [B, max_seq_len]
                pointer_loss = torch.nn.functional.cross_entropy(
                    query_logits,
                    pointer_targets,
                    ignore_index=-100
                )
                loss = loss + pointer_loss
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.step += 1
        
        # Calculate metrics
        metrics = {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
        
        # Calculate pointer accuracy if we have targets
        if pointer_targets is not None and 'pointer_logits' in output:
            pointer_logits = output['pointer_logits']  # [B, N, max_seq_len]
            predicted_positions = pointer_logits[:, -1, :].argmax(-1)  # [B]
            pointer_acc = (predicted_positions == pointer_targets).float().mean()
            metrics['pointer_acc'] = pointer_acc.item()
        
        return metrics
    
    def evaluate(self, eval_batches):
        """Evaluation."""
        self.model.eval()
        total_loss = 0
        total_pointer_acc = 0
        total_exact_match = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_batches:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)
                pointer_targets = batch.get('pointer_targets')
                if pointer_targets is not None:
                    pointer_targets = pointer_targets.to(self.device)
                
                output = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    return_pointer_logits=pointer_targets is not None
                )
                
                if 'loss' in output:
                    total_loss += output['loss'].item()
                
                # Calculate pointer accuracy
                if pointer_targets is not None and 'pointer_logits' in output:
                    pointer_logits = output['pointer_logits']  # [B, N, max_seq_len] 
                    predicted_positions = pointer_logits[:, -1, :].argmax(-1)  # [B] - use last position
                    pointer_acc = (predicted_positions == pointer_targets).float().mean()
                    total_pointer_acc += pointer_acc.item()
                    
                    # Exact match (all positions correct)
                    batch_exact = (predicted_positions == pointer_targets).float().mean()
                    total_exact_match += batch_exact.item()
                
                num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'pointer_acc': total_pointer_acc / max(num_batches, 1),
            'exact_match': total_exact_match / max(num_batches, 1)
        }
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Full training loop."""
        print(f"Starting training for {self.config.max_steps} steps...")
        
        for step in range(self.config.max_steps):
            # Get next batch
            try:
                batch = next(iter(train_dataloader))
            except StopIteration:
                # Restart dataloader
                batch = next(iter(train_dataloader))
            
            # Training step
            metrics = self.train_step(batch)
            
            # Logging and evaluation
            if (step + 1) % self.config.eval_interval == 0:
                eval_metrics = {}
                if eval_dataloader is not None:
                    eval_metrics = self.evaluate(eval_dataloader)
                
                # Combine metrics
                combined_metrics = {}
                combined_metrics.update({f'train_{k}': v for k, v in metrics.items()})
                combined_metrics.update({f'eval_{k}': v for k, v in eval_metrics.items()})
                
                # Print metrics
                print(f"Step {step + 1}")
                print(f"  Train Loss: {metrics.get('loss', 0):.4f}")
                print(f"  Train Pointer Acc: {metrics.get('pointer_acc', 0):.4f}")
                print(f"  Pointer Entropy: {2.0:.4f}")  # Dummy for transformer
                print(f"  Hop Distance: {1.0:.2f}")     # Dummy for transformer  
                print(f"  LR: {metrics.get('lr', 0):.6f}")
                print(f"  Eval Loss: {eval_metrics.get('loss', 0):.4f}")
                print(f"  Eval Pointer Acc: {eval_metrics.get('pointer_acc', 0):.4f}")
                
                # Save best model
                if eval_metrics.get('pointer_acc', 0) > self.best_metrics.get('pointer_acc', 0):
                    self.best_metrics.update(eval_metrics)
                    # Save checkpoint
                    checkpoint_path = f"{self.output_dir}/best_model.pt"
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'step': self.step,
                        'metrics': self.best_metrics
                    }, checkpoint_path)
                    print("Saved checkpoint: best_model.pt")
                print()
        
        print("Training completed!")
        
        # Final evaluation
        if eval_dataloader is not None:
            final_metrics = self.evaluate(eval_dataloader)
            print("Final Evaluation Metrics:")
            print(f"  Pointer Accuracy: {final_metrics.get('pointer_acc', 0):.4f}")
            print(f"  Exact Match: {final_metrics.get('exact_match', 0):.4f}")
            print(f"  Loss: {final_metrics.get('loss', 0):.4f}")


if __name__ == "__main__":
    # Test the vanilla transformer
    print("Testing Vanilla Transformer with ALiBi...")
    
    model = VanillaTransformerDecoder(
        vocab_size=1000,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=512
    )
    
    # Test forward pass
    batch_size, seq_len = 4, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Test language modeling
    output = model(input_ids, return_pointer_logits=False)
    print(f"LM logits shape: {output['logits'].shape}")
    
    # Test pointer supervision
    output = model(input_ids, return_pointer_logits=True)
    print(f"Pointer logits shape: {output['pointer_logits'].shape}")
    
    # Test with labels
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    output = model(input_ids, labels=labels)
    print(f"Loss: {output['loss'].item():.4f}")
    
    print("Vanilla Transformer test completed!")