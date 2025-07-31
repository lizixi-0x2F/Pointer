import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Token embedding layer that converts discrete token IDs to dense vectors.
    
    Args:
        vocab_size (int): Size of the vocabulary
        d (int): Dimension of the embedding vectors
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(self, vocab_size, d, dropout=0.0):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.weight = nn.Embedding(vocab_size, d)
        self.dropout = nn.Dropout(dropout)
        
        std = 0.02 / math.sqrt(d)
        nn.init.normal_(self.weight.weight, mean=0.0, std=std)
    
    def forward(self, ids):
        """
        Args:
            ids (torch.Tensor): Input token IDs of shape [B, N]
        
        Returns:
            torch.Tensor: Embeddings of shape [B, N, d]
        """
        ids = torch.clamp(ids, 0, self.vocab_size - 1)
        
        embeddings = self.weight(ids)
        embeddings = self.dropout(embeddings)
        
        return embeddings