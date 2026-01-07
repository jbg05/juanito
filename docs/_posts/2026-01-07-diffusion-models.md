---
title: "Diffusion Models: Mathematical Foundations"
date: 2026-01-07
categories: [notes]
tags: [ml, diffusion, generative-models, score-matching]
toc: true
toc_sticky: true
math: true
---

*Inspired by [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970), [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), and [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).*

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

A VDM is a Markovian HVAE with three restrictions: (1) latent dimension equals data dimension, (2) encoder structure at each timestep is fixed as a linear Gaussian, (3) Gaussian parameters vary over time such that $p(x_T)$ is standard Gaussian.

Notation: $x_t$ where $t=0$ is data, $t \in [1,T]$ are latents.

**Forward process:**

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})
$$

where encoder transitions are fixed:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)\mathbf{I})
$$

**Reverse process (joint):**

$$
p(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t), \quad p(x_T) = \mathcal{N}(0, \mathbf{I})
$$

**ELBO derivation.** Start from log evidence:

$$
\begin{align}
\log p(x) &= \log \int p(x_{0:T}) dx_{1:T} \\
&= \log \int \frac{p(x_{0:T}) q(x_{1:T} \mid x_0)}{q(x_{1:T} \mid x_0)} dx_{1:T} \\
&= \log \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \frac{p(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] \\
&\geq \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p(x_{0:T})}{q(x_{1:T} \mid x_0)} \right]
\end{align}
$$

Plug in joint and posterior:

$$
= \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p(x_T) p_\theta(x_0 \mid x_1) \prod_{t=2}^T p_\theta(x_{t-1} \mid x_t)}{\prod_{t=1}^T q(x_t \mid x_{t-1})} \right]
$$

Key insight: rewrite encoder $q(x_t \mid x_{t-1})$ using Bayes rule to condition on $x_0$:

$$
q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0) q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}
$$

After algebraic manipulations (see paper), we get:

$$
\begin{align}
\log p(x) &\geq \mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)] \\
&\quad - D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T)) \\
&\quad - \sum_{t=2}^T \mathbb{E}_{q(x_{t-1}, x_{t+1} \mid x_0)} \left[ D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t)) \right]
\end{align}
$$

Three terms:

1. **Reconstruction:** $\mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]$ - decode from first latent.

2. **Prior matching:** $D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T))$ - final latent should be Gaussian (typically zero with good schedule).

3. **Denoising matching:** $\sum_{t=2}^T \mathbb{E}_{q(x_{t-1}, x_{t+1} \mid x_0)} [D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))]$ - learned reverse step $p_\theta(x_{t-1} \mid x_t)$ matches ground-truth denoising step $q(x_{t-1} \mid x_t, x_0)$. Since $q(x_{t-1} \mid x_t, x_0)$ has access to clean $x_0$, it defines how to denoise $x_t$ given what $x_0$ should be.

Lower variance than standard HVAE derivation: each expectation is over at most one random variable at a time.

---

## References

Based on "Understanding Diffusion Models: A Unified Perspective" (arXiv:2208.11970) and related work.

