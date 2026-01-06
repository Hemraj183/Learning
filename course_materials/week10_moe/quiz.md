# Week 10 Quiz: Mixture of Experts

## Question 1: Sparsity
**What does it mean for an MoE model to be "Sparse"?**
- [ ] It has very few parameters
- [ ] For any given input, only a small subset of parameters (experts) are activated
- [ ] It uses sparse matrices
- [ ] It is trained on sparse data

<details>
<summary>Answer</summary>
**Only a small subset of parameters are activated**. This decouples model size from inference cost.
</details>

---

## Question 2: The Gating Network
**What is the job of the Gating Network?**
- [ ] To stop the model from overfitting
- [ ] To decide which Expert(s) process which token
- [ ] To normalize the inputs
- [ ] To add noise

<details>
<summary>Answer</summary>
**To decide which Expert(s) process which token**. It learns to specialize the experts.
</details>
