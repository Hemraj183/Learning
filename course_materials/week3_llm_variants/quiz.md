# Week 3 Quiz: LLM Variants & Latent Spaces

## Question 1: Byte-Pair Encoding (BPE)
**Why do modern LLMs use subword tokenization (like BPE) instead of character or word-level tokenization?**
- [ ] It reduces the vocabulary size compared to word-level, while avoiding the long sequences of character-level
- [ ] It handles "unknown" (OOV) words by breaking them into known subwords
- [ ] It is computationally efficient
- [ ] All of the above

<details>
<summary>Answer</summary>
**All of the above**. BPE strikes a balance. It keeps common words as single tokens (efficiency) but can represent any string by falling back to characters (no OOV).
</details>

---

## Question 2: GPT vs. BERT
**Which of the following is true about the architectural difference between GPT and BERT?**
- [ ] GPT is an Encoder-only model; BERT is Decoder-only
- [ ] GPT is a Decoder-only model (Auto-regressive); BERT is Encoder-only (Auto-encoding)
- [ ] They are identical, just trained differently
- [ ] GPT uses LSTM; BERT uses Transformers

<details>
<summary>Answer</summary>
**GPT is a Decoder-only model; BERT is Encoder-only**. GPT is designed for generation (predicting the next token), masking future tokens. BERT is designed for understanding (predicting masked tokens in the middle), seeing the full context.
</details>

---

## Question 3: Latent Space
**If two words have a high cosine similarity in the model's embedding layer, what does that imply?**
- [ ] They appear next to each other frequently in the training data
- [ ] They are semantically or syntactically similar (e.g., "Cat" and "Dog")
- [ ] They simplify to the same token ID
- [ ] The model is overfitting

<details>
<summary>Answer</summary>
**They are semantically or syntactically similar**. The embedding layer learns a geometric representation where similar concepts are close together in the vector space.
</details>

---

## Question 4: Temperature Sampling
**What happens when you increase the "temperature" parameter during text generation?**
- [ ] The model becomes more confident and repetitive (picks the most likely token)
- [ ] The distribution flattens, making the model more random and creative (but potentially incoherent)
- [ ] The model stops generating earlier
- [ ] The model switches from English to French

<details>
<summary>Answer</summary>
**The distribution flattens, making the model more random**. High temperature divides the logits, making the probabilities more uniform. Low temperature (near 0) approximates greedy search.
</details>
