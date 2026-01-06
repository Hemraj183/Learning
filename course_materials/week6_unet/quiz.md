# Week 6 Quiz: U-Net Architecture

## Question 1: Skip Connections
**What is the purpose of the skip connections (concatenation) in U-Net?**
- [ ] To bypass layers for faster training
- [ ] To preserve high-frequency spatial details lost during downsampling
- [ ] To reduce parameter count
- [ ] To increase the receptive field

<details>
<summary>Answer</summary>
**To preserve high-frequency spatial details**. They combine "what" (semantic, deep) with "where" (spatial, shallow).
</details>

---

## Question 2: Bottleneck
**What happens at the bottom (bottleneck) of the U-Net?**
- [ ] The image resolution is highest
- [ ] The image resolution is lowest, but feature depth is highest
- [ ] The model stops training
- [ ] Nothing

<details>
<summary>Answer</summary>
**Resolution is lowest, feature depth is highest**. This captures the most abstract semantic concepts.
</details>
