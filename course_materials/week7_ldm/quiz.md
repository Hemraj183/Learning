# Week 7 Quiz: Latent Diffusion

## Question 1: Latent Space
**Why do Latent Diffusion Models (LDM) run faster than Pixel Diffusion?**
- [ ] They use faster GPUs
- [ ] They operate in a compressed latent space (e.g., 64x64) instead of full pixel space (e.g., 512x512)
- [ ] They don't use U-Net
- [ ] They skip steps

<details>
<summary>Answer</summary>
**They operate in a compressed latent space**. This reduces the computational complexity significantly.
</details>

---

## Question 2: VAE
**What component is responsible for compressing images into latent space and reconstructing them?**
- [ ] CLIP Text Encoder
- [ ] U-Net
- [ ] Variational Autoencoder (VAE)
- [ ] Discriminator

<details>
<summary>Answer</summary>
**Variational Autoencoder (VAE)**. The VAE Encoder compresses, the VAE Decoder reconstructs.
</details>
