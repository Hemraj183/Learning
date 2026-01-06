# Week 1 Quiz: PyTorch Fundamentals

## Question 1: Tensors & Gradients
**What happens if you run the following code?**
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
z = x ** 3
z.backward()
print(x.grad)
```
- [ ] Prints 4.0
- [ ] Prints 12.0
- [ ] Prints 16.0
- [ ] Throws an error

<details>
<summary>Answer</summary>
**Prints 16.0**. Gradients accumulate! The first backward adds 4.0 (2x), the second adds 12.0 (3x^2). You must call `x.grad.zero_()` to reset.
</details>

---

## Question 2: nn.Parameter
**Why do we use `nn.Parameter` instead of a regular tensor inside `nn.Module`?**
- [ ] It makes the tensor immutable
- [ ] It automatically registers the tensor as a learnable parameter so `model.parameters()` can behave correctly
- [ ] It moves the tensor to GPU automatically
- [ ] There is no difference

<details>
<summary>Answer</summary>
**It automatically registers the tensor as a learnable parameter**. This ensures optimizers can find it and it's included in `state_dict`.
</details>

---

## Question 3: Computational Graph
**True or False: PyTorch builds a static computation graph before running the forward pass.**
- [ ] True
- [ ] False

<details>
<summary>Answer</summary>
**False**. PyTorch uses **Dynamic Computation Graphs** (define-by-run), meaning the graph is built on the fly as you execute operations.
</details>

---

## Question 4: Device Management
**Which of the following determines if two tensors can be added together?**
- [ ] They must be the same shape
- [ ] They must be on the same device (CPU/GPU)
- [ ] They must have the same dtype
- [ ] All of the above (mostly)

<details>
<summary>Answer</summary>
**All of the above**. But the most critical runtime error usually comes from mixing devices (e.g., adding a CPU tensor to a CUDA tensor).
</details>

---

## Question 5: Backpropagation
**In the context of `y = f(g(x))`, what does the chain rule state?**
- [ ] dy/dx = dy/dg + dg/dx
- [ ] dy/dx = dy/dg * dg/dx
- [ ] dy/dx = dy/dg - dg/dx

<details>
<summary>Answer</summary>
**dy/dx = dy/dg * dg/dx**. The derivative of the composite function is the product of the derivatives.
</details>
