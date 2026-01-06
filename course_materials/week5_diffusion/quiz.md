# Week 5 Quiz: Diffusion Math

## Question 1: The Forward Process
**What happens in the Forward Diffusion Process q(x_t | x_0)?**
- [ ] The image is denoised
- [ ] Gaussian noise is incrementally added to the image until it becomes pure noise
- [ ] The image is compressed
- [ ] The image is rotated

<details>
<summary>Answer</summary>
**Gaussian noise is incrementally added**. This destroys the data distribution.
</details>

---

## Question 2: The Reparameterization Trick
**Why do we write x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps?**
- [ ] To sample x_t at any timestep t in one go, without looping t times
- [ ] To make it differentiable
- [ ] To save memory
- [ ] It looks cool

<details>
<summary>Answer</summary>
**To sample x_t at any timestep t in one go**. This efficient property is crucial for training.
</details>
