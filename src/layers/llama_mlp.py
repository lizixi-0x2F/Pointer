"""
Llama-style SwiGLU MLP implementation for Pointer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LlamaMLP(nn.Module):
    """
    Llama-style SwiGLU MLP (Multi-Layer Perceptron)
    
    Uses SwiGLU activation instead of standard ReLU/GELU for better performance.
    Architecture: gate_proj -> silu -> up_proj -> elementwise_mul -> down_proj
    
    Args:
        hidden_size (int): Input/output hidden dimension
        intermediate_size (int): Intermediate (expanded) dimension  
        hidden_act (str): Activation function name (should be "silu" for SwiGLU)
        bias (bool): Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        
        # SwiGLU components
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        # Activation function
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "relu":
            self.act_fn = F.relu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU MLP
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, hidden_size]
        """
        # SwiGLU: gate * silu(up) -> down
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Apply activation to gate (SwiGLU pattern)
        activated_gate = self.act_fn(gate)
        
        # Element-wise multiplication (gating)
        hidden_states = activated_gate * up
        
        # Down projection
        output = self.down_proj(hidden_states)
        
        return output


class LlamaGLU(nn.Module):
    """
    Alternative implementation using explicit GLU formulation
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Single linear layer that outputs 2 * intermediate_size
        # Will be split into gate and up components
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "relu":
            self.act_fn = F.relu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GLU MLP
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, hidden_size]
        """
        # Project to 2 * intermediate_size
        gate_up = self.gate_up_proj(x)
        
        # Split into gate and up components
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Apply GLU: gate * activation(up)
        hidden_states = self.act_fn(gate) * up
        
        # Down projection
        output = self.down_proj(hidden_states)
        
        return output