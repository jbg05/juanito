---
title: "Diffusion Models: Mathematical Foundations"
date: 2026-01-07
categories: [notes]
tags: [ml, diffusion, generative-models, score-matching]
toc: true
toc_sticky: true
math: true
---

## Background: ELBO and VAE

Say we're modeling data $x$ using some latent variable $z$ through joint $p(x,z)$. Want to compute $p(x)$? Two options:

$$
p(x) = \int p(x,z) dz \quad \text{or} \quad p(x) = \frac{p(x,z)}{p(z \mid x)}
$$

Problem is both suck. First one means integrating over all possible latents. Second one needs the true posterior $p(z \mid x)$ which we don't have.

**Getting the ELBO.** Here's the workaround: introduce an approximate posterior $q_\phi(z \mid x)$ that we *can* work with. Now we can derive a lower bound on the log evidence.

Via Jensen's inequality:

$$
\begin{align}
\log p(x) &= \log \int p(x,z) dz \\
&= \log \int \frac{p(x,z) q_\phi(z \mid x)}{q_\phi(z \mid x)} dz \\
&= \log \mathbb{E}_{q_\phi(z \mid x)} \left[ \frac{p(x,z)}{q_\phi(z \mid x)} \right] \\
&\geq \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right]
\end{align}
$$

Or from the chain rule angle:

$$
\begin{align}
\log p(x) &= \mathbb{E}_{q_\phi(z \mid x)}[\log p(x)] \\
&= \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{p(z \mid x)} \right] \\
&= \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z) q_\phi(z \mid x)}{p(z \mid x) q_\phi(z \mid x)} \right] \\
&= \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right] + D_{\text{KL}}(q_\phi(z \mid x) \| p(z \mid x))
\end{align}
$$

KL divergence is always $\geq 0$, so we've got ourselves a legit lower bound. The tighter our $q_\phi$ approximates the true posterior, the closer we get to the actual evidence.

**VAE breakdown.** Let's split up the ELBO into something more intuitive:

$$
\mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right] = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} \right]
= \underbrace{\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \| p(z))}_{\text{prior matching}}
$$

The encoder $q_\phi(z \mid x)$ compresses $x$ into a distribution over latents. The decoder $p_\theta(x \mid z)$ tries to reconstruct the original $x$ from those latents. Standard setup uses Gaussians:

$$
q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) \mathbf{I}), \quad p(z) = \mathcal{N}(0, \mathbf{I})
$$

**Reparameterization trick.** Can't backprop through sampling, so we rewrite: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. Now the randomness is separated out and we can differentiate through $\mu_\phi$ and $\sigma_\phi$. KL term becomes closed-form, reconstruction gets Monte Carlo'd:

$$
\max_{\phi,\theta} \sum_{l=1}^L \log p_\theta(x \mid z^{(l)}) - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))
$$

### Hierarchical VAE (Markovian)

Instead of one latent, chain together $T$ of them in a Markov structure. Joint distribution and posterior look like:

$$
p(x, z_{1:T}) = p(z_T) p_\theta(x \mid z_1) \prod_{t=2}^T p_\theta(z_{t-1} \mid z_t)
$$

$$
q_\phi(z_{1:T} \mid x) = q_\phi(z_1 \mid x) \prod_{t=2}^T q_\phi(z_t \mid z_{t-1})
$$

ELBO extends naturally:

$$
\mathbb{E}_{q_\phi(z_{1:T} \mid x)} \left[ \log \frac{p(x, z_{1:T})}{q_\phi(z_{1:T} \mid x)} \right] = \mathbb{E}_{q_\phi} \left[ \log \frac{p(z_T) p_\theta(x \mid z_1) \prod_{t=2}^T p_\theta(z_{t-1} \mid z_t)}{q_\phi(z_1 \mid x) \prod_{t=2}^T q_\phi(z_t \mid z_{t-1})} \right]
$$

---

## Variational Diffusion Models

Here's where it gets cool. A **Variational Diffusion Model (VDM)** is basically just a Markovian HVAE with three specific choices that make everything work beautifully:

1. **Latent dimension = data dimension.** No compression, latents $x_t$ have same shape as data $x_0$.

2. **Encoder structure is fixed**, not learned. At each timestep, the encoder is a linear Gaussian centered at the previous timestep's output. Meaning we're just progressively adding noise in a predetermined way.

3. **Variance schedule.** The Gaussian parameters evolve over time so that by the final timestep $T$, the distribution $p(x_T)$ is pure standard Gaussian noise.

Since latent dimension matches data dimension, we can use unified notation: $x_t$ where $t=0$ is real data and $t \in [1,T]$ are noisy latent versions indexed by hierarchy depth.

**Forward process (encoder).** The forward transitions are fixed Gaussians:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)\mathbf{I})
$$

where $\alpha_t$ is a learnable (or fixed) coefficient controlling variance preservation. Different parameterizations work, but the key idea is we're progressively noisifying while keeping variance scale consistent. Full forward chain:

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})
$$

**Reverse process (decoder).** We learn $p_\theta(x_{t-1} \mid x_t)$ to denoise. The decoder reverses the noise:

$$
p(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
$$

where $p(x_T) = \mathcal{N}(0, \mathbf{I})$ is standard Gaussian.

**The ELBO.** Just like any HVAE, we maximize the ELBO. But here's the clever part: we can derive it in a way that reduces variance and gives better intuition. Starting from the log evidence and using Bayes rule on the encoder:

$$
q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0) q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}
$$

After some algebra (equations 34-45 in your screenshots), the ELBO decomposes into three interpretable terms:

$$
\log p(x) \geq \underbrace{\mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T))}_{\text{prior matching}} - \sum_{t=2}^T \underbrace{\mathbb{E}_{q(x_{t-1}, x_{t+1} \mid x_0)}[D_{\text{KL}}(q(x_t \mid x_{t-1}, x_0) \| p_\theta(x_{t-1} \mid x_t))]}_{\text{denoising matching}}
$$

**What each term does:**

1. **Reconstruction term** - How well can we reconstruct $x_0$ from the first latent $x_1$? Just like vanilla VAE.

2. **Prior matching term** - How close is our final noised state $q(x_T \mid x_0)$ to pure Gaussian $p(x_T)$? This is typically zero if we choose a good schedule.

3. **Denoising matching term** - Here's the magic. For every intermediate timestep $t$, we're learning to denoise: the learned backward step $p_\theta(x_{t-1} \mid x_t)$ should match the tractable forward denoising step $q(x_{t-1} \mid x_t, x_0)$. The $q(x_{t-1} \mid x_t, x_0)$ acts as ground truth because it knows what the clean image $x_0$ looks like.

This formulation has lower variance than the naive HVAE derivation because each expectation is over at most one random variable at a time, instead of two.

**To sample:** Start with pure noise $x_T \sim \mathcal{N}(0, \mathbf{I})$ and iteratively denoise using learned transitions $p_\theta(x_{t-1} \mid x_t)$ for $T$ steps to generate a novel $x_0$.

---

## References

Based on "Understanding Diffusion Models: A Unified Perspective" (arXiv:2208.11970) and related work.

