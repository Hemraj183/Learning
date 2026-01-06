# Week 2 Quiz: The Transformer Architecture

## Question 1: Self-Attention Complexity
**What is the time complexity of the Scaled Dot-Product Attention mechanism for a sequence of length $N$ and embedding dimension $d$?**
- [ ] $O(N)$
- [ ] $O(N^2 d)$
- [ ] $O(Nd^2)$
- [ ] $O(N \log N)$

<details>
<summary>Answer</summary>
**$O(N^2 d)$**. Computing the attention scores requires multiplying a $N \times d$ matrix (Q) by a $d \times N$ matrix ($K^T$), resulting in an $N \times N$ matrix. This quadratic complexity is the bottleneck for long sequences.
</details>

---

## Question 2: Positional Encoding
**Why do Transformers need Positional Encodings?**
- [ ] Because the self-attention mechanism is permutation invariant (it has no notion of order)
- [ ] To increase the capacity of the model
- [ ] To help the model converge faster
- [ ] They don't need them, it's just a convention

<details>
<summary>Answer</summary>
**Because the self-attention mechanism is permutation invariant**. Unlike RNNs which process tokens sequentially, attention processes all tokens in parallel. Without positional encodings, "The dog bit the man" and "The man bit the dog" would look identical to the self-attention layer.
</details>

---

## Question 3: Multi-Head Attention
**What is the benefit of Multi-Head Attention over a single attention head?**
- [ ] It's faster to compute
- [ ] It allows the model to attend to information from different representation subspaces at different positions
- [ ] It reduces the number of parameters
- [ ] It prevents overfitting

<details>
<summary>Answer</summary>
**It allows the model to attend to information from different representation subspaces at different positions**. One head might focus on syntactic relationships (subject-verb), while another focuses on semantic ones (entity linkage).
</details>

---

## Question 4: Decoder Masking
**In a GPT-style Decoder (Autoregressive), why do we mask the future tokens?**
- [ ] To prevent the model from cheating by seeing the answer (next token)
- [ ] To save memory
- [ ] Because future tokens don't exist yet during training
- [ ] To make training harder

<details>
<summary>Answer</summary>
**To prevent the model from cheating**. During training, we feed the entire sequence at once (parallelization). If we didn't mask position $i$ from attending to $i+1$, the model would just copy the next token instead of learning to predict it.
</details>
