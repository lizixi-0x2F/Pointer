"""
Losses module for Pointer Architecture

专门的损失函数模块，包含基于指针的反思损失
"""

from .reflection_loss import PointerReflectionLoss, create_reflection_loss

__all__ = [
    'PointerReflectionLoss',
    'create_reflection_loss'
]