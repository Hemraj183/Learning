# Week 9 Quiz: LoRA

## Question 1: Rank
**In LoRA, what does the rank 'r' represent?**
- [ ] The quality of the model
- [ ] The dimension of the low-rank bottleneck matrices (A and B)
- [ ] The learning rate
- [ ] The sequence length

<details>
<summary>Answer</summary>
**The dimension of the bottleneck**. A lower rank means fewer trainable parameters (more efficiency).
</details>

---

## Question 2: Frozen Weights
**During LoRA fine-tuning, what happens to the original model weights?**
- [ ] They are updated via gradient descent
- [ ] They are deleted
- [ ] They remain frozen (immutable)
- [ ] They are quantized to 1-bit

<details>
<summary>Answer</summary>
**They remain frozen**. Only the small LoRA adapter matrices are updated.
</details>
