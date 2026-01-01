# Week 2: The Transformer Architecture (The Engine) 🎯

Welcome to Week 2 of the AI Learning Journey! This week, we dive deep into the heart of modern AI: the **Transformer Architecture**.

## 🎯 Learning Objectives

- **Self-Attention mechanism**: Understand how tokens "attend" to each other.
- **Multi-Head Attention**: Learn why multiple attention heads are better than one.
- **Positional Encoding**: Discover how Transformers understand word order without recurrence.
- **Q, K, V Mechanism**: Develop an intuitive understanding of Queries, Keys, and Values.

## 📖 Theory & Resources

- **Primary Paper**: ["Attention Is All You Need" (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- **Top Resource**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- **Video**: [The Transformer - 3Blue1Brown](https://www.youtube.com/watch?v=wjZofJX0v4M) (if available)

## 🛠️ Coding Exercise

**Task**: Implement a single `TransformerBlock` class from scratch in PyTorch.

**Architecture**:
1. LayerNorm
2. Self-Attention (or Multi-Head Attention)
3. LayerNorm
4. Feed-Forward Network (Linear -> ReLU -> Linear)

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # Your implementation here
        pass

    def forward(self, value, key, query, mask):
        # Your implementation here
        pass
```

## 🏁 Milestone

You'll know you're ready to move on when you can explain exactly how the Attention mechanism replaces Recurrence and how information flows through the network.

---
*Stay curious and keep building!* 🚀
