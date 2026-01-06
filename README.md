# Deep Learning Mastery: 12-Week Interactive Journey 🚀

> **A comprehensive curriculum for mastering Tensors, Transformers, Diffusion, and MoE.**

[![GitHub](https://img.shields.io/badge/GitHub-Hemraj183-blue?logo=github)](https://github.com/Hemraj183)
[![Platform-Premium](https://img.shields.io/badge/UI-Premium%20V2-purple.svg)]()
[![Progress](https://img.shields.io/badge/Curriculum-12%20Weeks%20Complete-green.svg)]()

## 🌟 The Interactive Platform
This repository now features a **Premium Interactive Learning Platform**. 
Launch it by opening:
👉 **[`interactive_platform/index.html`](./interactive_platform/index.html)**

### Platform Features:
- 🎨 **Premium UI**: Sleek dark mode with glassmorphism and vibrant gradients.
- 📈 **Progress Tracking**: Persistent module-level and global progress tracking using `localStorage`.
- 🧩 **Interactive Labs**: Custom-built visualizations for every week (Computational Graphs, Noise Schedulers, Matrix Decompositions).
- 💻 **Code implementation**: Direct access to core logic and step-by-step project guides.
- 🗺️ **Global Navigation**: Seamlessly switch between weeks with the unified sidebar.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Hemraj183/Learning.git
cd Learning
```

### 2. Install Dependencies
To run the code implementations locally (not required for the interactive web UI), install the Python requirements:
```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Platform
The easiest way to explore the curriculum is to open the local dashboard:
- Simply double-click **[`interactive_platform/index.html`](./interactive_platform/index.html)** in your file explorer.
- No local server (Node.js/Python) is required—it runs entirely in your browser using Vanilla JS.

### 4. Regenerate the Site (Optional)
If you modify the curriculum data in `generate_website_v2.py`, you can regenerate the entire platform:
```bash
python generate_website_v2.py
```

---

## � Global Access (Host it Everywhere)
You can host this platform for **FREE** using GitHub Pages so you can access it from your phone, tablet, or any computer:

1. Go to your repository on GitHub: **[Hemraj183/Learning](https://github.com/Hemraj183/Learning)**
2. Click on **Settings** (top tab).
3. Click on **Pages** (left sidebar).
4. Under **Branch**, select `master` and the folder `/(root)`.
5. Click **Save**.
6. GitHub will provide a link like: `https://hemraj183.github.io/Learning/`

---

## �🎯 The 12-Week Architecture

### Month 1: LLM Foundations 🧠
1. **Week 1: PyTorch Foundations** - Tensors, Autograd, and building MLPs from scratch.
2. **Week 2: Transformers** - Self-Attention, Multi-Head Attention, and LayerNorm.
3. **Week 3: LLM Variants** - GPT vs BERT, Tokenization, and Latent Spaces.
4. **Week 4: The Router** - Building classification systems for model dispatching.

### Month 2: Generative AI 🎨
5. **Week 5: Diffusion Math** - Forward/Reverse processes and the Reparameterization Trick.
6. **Week 6: U-Net Architecture** - Skip connections and spatial feature persistence.
7. **Week 7: Latent Diffusion** - VAE compression and Stable Diffusion (LDM).
8. **Week 8: Noisy Router** - Multi-modal agent routing for Text vs Image.

### Month 3: Optimization & Scale 🚀
9. **Week 9: LoRA** - Low-Rank Adaptation for efficient fine-tuning.
10. **Week 10: MoE** - Mixture of Experts and sparse gating networks.
11. **Week 11: Optimization** - Quantization (INT8), Memory profiling, and KV-Cache.
12. **Week 12: Final Capstone** - The Ultimate Assistant (Integrated MoE + LoRA + Router).

---

## 📂 Repository Structure

```
Learning/
├── interactive_platform/      # 🏠 The Premium Learning Hub
│   ├── index.html             # 🛰️ Course Dashboard
│   └── modules/               # 📦 All 12 Learning Modules
│       ├── week1_pytorch/
│       ├── week2_transformer/
│       └── ... (Weeks 3-12)
├── generate_website_v2.py     # ⚙️ The Python Generator (V2 Premium Engine)
├── course_materials/          # 📚 Raw templates and assets
└── README.md                  # This file
```

## 🛠️ Build System
The entire platform is generated using a custom Python engine:
- `generate_website_v2.py`: Generates the HTML5/CSS3/JS core for all modules.
- **Vanilla Tech Stack**: No heavy frameworks—just pure performance and clean visuals.

## 🎓 Completion Status

| Week | Phase | Topic | Status |
|------|-------|-------|--------|
| 1-4 | Foundations | LLM & Routing | ✅ Complete |
| 5-8 | Creative | Diffusion & LDMs | ✅ Complete |
| 9-12| Advanced | LoRA, MoE & Opt | ✅ Complete |

---

**Last Updated**: January 6, 2026  
**Curriculum Version**: 2.1 (Premium)

> "The journey of a thousand miles begins with a single step." - Lao Tzu

**Happy Learning! 🚀**

