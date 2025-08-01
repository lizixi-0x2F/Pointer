# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is a **Pointer Network** implementation featuring a novel neural architecture designed for enhanced long-range reasoning and multi-hop composition tasks. The architecture implements learnable address selection with layer-wise pointer chaining.

### Core Components

- **PointerDecoder** (`src/model/pointer_model.py`): Main model implementing DeepSeek-style architecture with token embedding, multi-layer pointer layers, RMSNorm, and language modeling head
- **PointerLayer** (`src/layers/pointer_layer.py`): Core layer with pre-norm architecture (RMSNorm → PointerBlock → Residual → RMSNorm → SwiGLU → Residual)
- **PointerBlock** (`src/layers/pointer_block.py`): Generates sparse address distributions, performs top-k selection, and aggregates neighbor vectors using AliBi positional encoding and GQA support
- **Reflection Mechanism**: Specialized layers that can backtrack through previous layer history to enhance reasoning capabilities

### Key Features

1. **Pointer Chaining**: Uses `prev_idx` to form pointer chains across layers, enabling multi-hop reasoning
2. **Reflection Layers**: Configurable layers that aggregate historical information from previous layers with learnable reflection gates
3. **GQA Support**: Grouped-Query Attention compatibility with separate key-value heads
4. **AliBi Positional Encoding**: For better length extrapolation
5. **Specialized Loss Function**: `PointerReflectionLoss` that combines language modeling loss with reflection-specific losses (consistency, selection, and reflection losses)

### Architecture Flow

```
Input Tokens → TokenEmbedding → PointerLayer(1..N) → RMSNorm → LM Head → Output
                                      ↓
                                PointerBlock (top-k selection + aggregation)
                                      ↓
                                SwiGLU FFN (Llama-style)
```

## Configuration

The model is configured through the `PointerDecoder` constructor parameters:

- `d`: Hidden dimension
- `n_layers`: Number of pointer layers
- `n_heads`: Number of attention heads
- `n_kv_heads`: Number of key-value heads (for GQA)
- `top_k`: Number of top positions selected by pointer mechanism
- `reflection_config`: Dictionary controlling reflection behavior
  - `reflection_layers`: List of layer indices that use reflection
  - `pointer_backtrack_layers`: Number of previous layers to consider for reflection
  - `reflection_gate_init`: Initial value for reflection gates

## Development Notes

### Python Environment
This is a pure PyTorch implementation with no specific build system detected. Key dependencies include:
- PyTorch (with CUDA support recommended)
- Standard scientific Python stack (numpy, etc.)

### Training and Testing
No specific test commands found in configuration files. The codebase appears to be research-focused with:
- Support for gradient checkpointing
- FP16/BF16 training compatibility
- KV caching for efficient inference
- Distillation support through hidden state outputs

### Code Organization
```
src/
├── layers/           # Neural network components
│   ├── pointer_block.py      # Core pointer selection mechanism
│   ├── pointer_layer.py      # Full pointer layer with FFN
│   ├── rmsnorm.py           # RMSNorm normalization
│   ├── alibi.py             # AliBi positional encoding
│   ├── embedding.py         # Token embeddings
│   └── llama_mlp.py         # SwiGLU FFN implementation
├── losses/           # Loss functions
│   └── reflection_loss.py    # Specialized reflection loss
└── model/            # Main model definitions
    └── pointer_model.py      # PointerDecoder main class
```

### Key Implementation Details

- Uses pre-norm architecture (normalize before transformation) following DeepSeek style
- Implements batched gathering for efficient top-k value aggregation
- Supports both training and inference modes with KV caching
- Pointer indices are chained across layers using `prev_idx` parameter
- Reflection mechanism stores and processes layer history for enhanced reasoning

### Memory Management
- Implements `PointerCache` for efficient KV caching during inference
- Gradient checkpointing support for memory-efficient training
- Automatic cache expansion when sequence length exceeds initial allocation

This architecture is particularly suited for tasks requiring:
- Long-range dependency modeling
- Multi-hop reasoning (A→B→C chains)
- Structured information retrieval
- Length extrapolation beyond training context