# PyTorch Mastery - Interactive Tutorial

> **A comprehensive, interactive web-based tutorial to master Deep Learning and PyTorch fundamentals**

[![GitHub](https://img.shields.io/badge/GitHub-Hemraj183%2FLearning-blue?logo=github)](https://github.com/Hemraj183/Learning)
[![License](https://img.shields.io/badge/License-Educational-green.svg)]()

## 🌟 Overview

An immersive learning experience featuring a **premium dark-mode design** with glassmorphism effects, **interactive visualizations**, and **hands-on coding exercises**. This tutorial will take you from PyTorch basics to building a complete Multi-Layer Perceptron from scratch.

## ✨ Key Features

### 🎨 Premium User Interface
- **Dark Mode Theme** with vibrant purple/cyan gradients
- **Glassmorphism Effects** on cards and panels with backdrop blur
- **Animated Background** with floating gradient orbs
- **Smooth Animations** for all interactions and transitions
- **Responsive Layout** that works on desktop, tablet, and mobile

### ⚡ Interactive Learning
- **20+ Runnable Code Examples** with syntax highlighting
- **Copy-to-Clipboard** functionality for all code snippets
- **Interactive Visualizations**:
  - Computational graph animations
  - Gradient flow demonstrations
  - Loss landscape with gradient descent path
- **Progress Tracking** that saves your completion status locally
- **Keyboard Shortcuts** (Alt + Arrow Keys to navigate)

### 📚 Comprehensive Content
- **8 Tutorial Sections** covering all PyTorch fundamentals
- **Complete MLP Project** for MNIST classification (>95% accuracy)
- **4 Coding Exercises** with solutions
- **8-Point Milestone Checklist** to track your mastery

## 🎯 What You'll Learn

- ✅ **Autograd**: Automatic differentiation and gradient computation
- ✅ **Custom nn.Modules**: Building neural network components from scratch using `nn.Parameter`
- ✅ **GPU Tensor Handling**: Efficient computation with CUDA
- ✅ **Backpropagation**: Understanding gradient flow and the chain rule
- ✅ **Optimizers**: Using AdamW and learning rate scheduling
- ✅ **Complete Project**: Build an MLP from scratch for MNIST (784 → 256 → 128 → 10)

## 🚀 Getting Started

### Prerequisites

```bash
# Install PyTorch and torchvision
pip install torch torchvision
```

### Running the Tutorial

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hemraj183/Learning.git
   cd Learning/pytorch-tutorial
   ```

2. **Open in browser**
   - Simply open `index.html` in your web browser
   - No server required - runs entirely client-side!

3. **Start learning**
   - Follow the sections in order
   - Copy code examples and run them locally
   - Check off milestones as you progress

## 📖 Tutorial Structure

### 1. Introduction to PyTorch
- What is PyTorch and why use it
- Creating your first tensor
- Basic tensor operations
- Interactive code examples

### 2. Autograd Deep Dive
- How automatic differentiation works
- **Interactive computational graph visualization**
- Gradient computation examples
- Common pitfalls (gradient accumulation)

### 3. Custom nn.Modules
- Building blocks of PyTorch models
- Creating custom layers with `nn.Parameter`
- Comparison: `nn.Linear` vs manual implementation
- Multi-layer module composition

### 4. GPU Tensor Handling
- Moving tensors to GPU
- Device management best practices
- Moving entire models to GPU
- Memory management tips

### 5. Backpropagation Theory
- Visual explanation of the chain rule
- **Interactive gradient flow visualization**
- Manual backpropagation examples
- **Loss landscape visualization** with gradient descent path

### 6. Optimizers (AdamW)
- How optimizers work
- Comparison of SGD, Adam, and AdamW
- Using AdamW in practice
- Learning rate scheduling

### 7. 🎯 MLP Project (Main Project)
**Build a complete MNIST classifier from scratch!**

- Implementation without `nn.Linear` (using `nn.Parameter`)
- Step-by-step guide:
  1. Manual Linear Layer with Xavier initialization
  2. MLP Architecture (784 → 256 → 128 → 10)
  3. Data Loading with torchvision
  4. Training Loop with AdamW
  5. Evaluation and metrics
  6. Complete training script

**Expected Results:**
- Training accuracy: ~99%
- Test accuracy: ~97-98%
- Training time: ~2-3 minutes on GPU

### 8. Exercises & Challenges
- **Exercise 1**: Gradient debugging
- **Exercise 2**: Adding dropout layers
- **Exercise 3**: Batch normalization
- **Challenge**: Implement He initialization

## 🎓 Learning Outcomes

After completing this tutorial, you will be able to:

- ✅ Create and manipulate PyTorch tensors
- ✅ Understand autograd and compute gradients
- ✅ Build custom nn.Modules with nn.Parameter
- ✅ Move models and tensors to GPU
- ✅ Understand backpropagation and the chain rule
- ✅ Use optimizers (especially AdamW)
- ✅ **Write a complete training loop without documentation**
- ✅ Build an MLP from scratch for MNIST

## 🖼️ Screenshots

### Interactive Tutorial Interface
The tutorial features a modern, premium design with smooth animations and interactive elements.

### Computational Graph Visualization
Interactive animations demonstrate how gradients flow through the network during backpropagation.

### Complete MLP Project
Step-by-step guide to building a Multi-Layer Perceptron from scratch with full code examples.

## 🎨 Design & Technology

**Built with modern web technologies:**
- **HTML5**: Semantic structure
- **CSS3**: Custom properties, glassmorphism, animations
- **Vanilla JavaScript**: No dependencies, lightweight and fast
- **Google Fonts**: Inter (UI) and Fira Code (code blocks)

**Design Principles:**
- Mobile-first responsive design
- High contrast for readability
- Intuitive navigation
- Progressive disclosure
- Accessibility-friendly

## 💡 Usage Tips

1. **Navigation**: Use the sidebar to jump between sections
2. **Progress**: Check off milestones as you complete them
3. **Code**: Click "Copy" to copy code snippets to clipboard
4. **Practice**: Run all examples in your local Python environment
5. **Exercises**: Complete the challenges to reinforce learning
6. **Shortcuts**: Use `Alt + Arrow Keys` to navigate sections

## 📂 Project Structure

```
pytorch-tutorial/
├── index.html      # Main tutorial page with all content
├── style.css       # Premium design system with animations
├── script.js       # Interactive features and visualizations
└── README.md       # This file
```

## 🔧 Technical Highlights

- **Lightweight**: No heavy frameworks, fast loading
- **Performant**: Smooth 60fps animations
- **Persistent**: Progress saved to localStorage
- **Accessible**: Semantic HTML and keyboard navigation
- **Responsive**: Works on all screen sizes

## 🚀 Next Steps

After mastering these fundamentals, explore:
- **CNNs** (Convolutional Neural Networks) for computer vision
- **RNNs/LSTMs** for sequence modeling
- **Transformers** for NLP and modern architectures
- **Advanced topics**: Transfer learning, fine-tuning, deployment

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to:
- Report issues
- Suggest new features
- Improve documentation
- Add more exercises

## 📝 License

This tutorial is free to use for educational purposes.

## 🙏 Acknowledgments

Built with ❤️ for PyTorch learners worldwide.

---

**Ready to become a PyTorch expert? Open `index.html` and start learning! 🚀**

**Repository**: [https://github.com/Hemraj183/Learning](https://github.com/Hemraj183/Learning)
