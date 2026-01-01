# Learning Repository 🚀

> **My journey to master Deep Learning, LLMs, and Diffusion Models**

[![GitHub](https://img.shields.io/badge/GitHub-Hemraj183-blue?logo=github)](https://github.com/Hemraj183)
[![Learning](https://img.shields.io/badge/Status-Active%20Learning-green.svg)]()

## 🎯 Learning Goals

This repository documents my journey to become proficient in modern AI/ML technologies with the ultimate goal of **creating new models from scratch**.

### Primary Objectives

1. ✅ **Deep Learning Fundamentals** - Master PyTorch and neural network basics
2. 🎯 **Large Language Models (LLMs)** - Understand transformer architecture and build LLMs from scratch
3. 🎯 **Diffusion Models** - Learn generative AI and create custom diffusion models
4. 🎯 **Advanced Research Foundations** - Explore PEFT, MoE, and optimization for the next generation of models

## 📚 12-Week Learning Roadmap

### Month 1: LLM Foundations 🎯

#### Week 1: PyTorch Mastery ✅
- Autograd, custom modules, GPU handling, and optimizers.
- **Project**: Built an MLP from scratch for MNIST (97-98% accuracy).

#### Week 2: The Transformer Architecture (The Engine) 🎯
- **Focus**: Self-Attention mechanism, Multi-Head Attention, and Positional Encoding.
- **Theory**: "Attention Is All You Need" (Vaswani et al.).
- **Coding**: Implement a `TransformerBlock` from scratch.

#### Week 3: Variants & Latent Spaces 🎯
- **Focus**: Encoder-Decoder (T5) vs. Decoder-only (GPT). BERT vs. GPT.
- **Theory**: Tokenization, Embeddings, and the "Latent Space".
- **Coding**: Load GPT-2 with Hugging Face, inspect tensor shapes (B, S, H).

#### Week 4: Month 1 Capstone – A "Router" Prototype 🎯
- **Goal**: Build a toy version of a "Planner Network".
- **Project**: Classifier to route prompts to specific modules using CLIP text embeddings.
- **Milestone**: Working script that converts text to features and makes routing decisions.

---

### Month 2: Generative AI & Diffusion Models 🎯

#### Week 5: The Math of Diffusion (Forward & Reverse) 🎯
- **Focus**: Forward noising process and reverse denoising process.
- **Theory**: DDPM (Ho et al.) and the simplified objective function.
- **Coding**: Implement and visualize the forward diffusion scheduler.

#### Week 6: The U-Net Architecture 🎯
- **Focus**: Residual connections, Down/Up-sampling, and Skip Connections.
- **Theory**: How models process spatial vs. semantic information.
- **Coding**: Build a "Mini-U-Net" to denoise small images (16x16) or signals.

#### Week 7: Latent Diffusion & Stable Diffusion (LDM) 🎯
- **Focus**: VAE and working in Latent Space. Cross-Attention mechanism.
- **Theory**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al.).
- **Coding**: Hook into Stable Diffusion v1.5 U-Net pass and profile tensor shapes.

#### Week 8: Month 2 Capstone – The "Noisy" Router 🎯
- **Project**: Train a tiny diffusion model (MNIST) with conditional switching.
- **Goal**: Build a "Planner" that decides which expert model to load based on input.

---

### Month 3: Optimization, Modularity & Advanced Research 🎯

#### Week 9: Parameter-Efficient Fine-Tuning (PEFT) & LoRA 🎯
- **Focus**: LoRA (Low-Rank Adaptation) math and matrices (W = W0 + BA).
- **Coding**: Train a LoRA adapter for Stable Diffusion on a specific style.

#### Week 10: Mixture of Experts (MoE) & Dynamic Routing 🎯
- **Focus**: Sparse activation and Load Balancing Loss.
- **Theory**: Switch Transformers and dynamic expert selection.
- **Coding**: Implement a simple "Switch Layer" in PyTorch.

#### Week 11: Inference Optimization & Memory Profiling 🎯
- **Focus**: VRAM management (Weights, KV Cache, Activations).
- **Theory**: Gradient Checkpointing and Model Offloading.
- **Coding**: Profile SD inference and implement memory-saving techniques.

#### Week 12: Month 3 Capstone – The "LoRA Router" 🎯
- **Goal**: Create a lightweight classifier to dynamically inject LoRA adapters.
- **Project**: Complete PoC for dynamic expert routing with memory logging.

---

## 📂 Repository Structure

```
Learning/
├── index.html                # 🏠 Central Learning Portal
├── pytorch-tutorial/         # ✅ Week 1: PyTorch Fundamentals
├── llm-foundations/          # 🎯 Week 2-4: LLM Foundations
├── diffusion-models/         # 🎯 Week 5-8: Generative AI (Coming Soon)
├── optimization-research/    # 🎯 Week 9-12: Advanced Topics (Coming Soon)
└── README.md                 # This file
```

## 🛠️ Technologies & Tools

- **Core**: PyTorch, TorchVision, Hugging Face Diffusers/Transformers
- **Research**: LoRA, PEFT, MoE, CLIP
- **Tools**: Git, Jupyter, Pyodide (for interactive web elements)

## 🎓 Progress Tracking

| Week | Topic | Status | Completion |
|------|-------|--------|------------|
| 1 | PyTorch Fundamentals | ✅ Complete | 100% |
| 2 | Transformer Architecture | ⚡ In Progress | 20% |
| 3 | LLM Variants & HF | 🎯 Planned | 0% |
| 4 | Router Capstone | 🎯 Planned | 0% |
| 5-8 | Diffusion Models | 🎯 Planned | 0% |
| 9-12| Research & Optimization| 🎯 Planned | 0% |

## 💡 Learning Principles

1. **Learn by Doing**: Implement everything from scratch before using libraries.
2. **Understand the Math**: Focus on the objective functions and matrix operations.
3. **Research First**: Read the foundational papers (Attention, DDPM, LoRA).
4. **Modular Design**: Build components that can be reused and combined.

---

**Last Updated**: January 1, 2026  
**Current Focus**: Week 2 - Transformer Architecture

> "The journey of a thousand miles begins with a single step." - Lao Tzu

**Let's build something amazing! 🚀**
