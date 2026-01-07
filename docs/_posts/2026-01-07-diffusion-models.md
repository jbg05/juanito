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

We model data $x$ with latent variable $z$ via joint distribution $p(x,z)$. Two ways to recover $p(x)$:

$$
p(x) = \int p(x,z) dz \quad \text{or} \quad p(x) = \frac{p(x,z)}{p(z \mid x)}
$$

Both intractable: first requires marginalizing all $z$, second needs true posterior $p(z \mid x)$.

**ELBO derivation.** Introduce approximate posterior $q_\phi(z \mid x)$. Start with log evidence:

$$
\begin{align}
\log p(x) &= \log \int p(x,z) dz \\
&= \log \int \frac{p(x,z) q_\phi(z \mid x)}{q_\phi(z \mid x)} dz \\
&= \log \mathbb{E}_{q_\phi(z \mid x)} \left[ \frac{p(x,z)}{q_\phi(z \mid x)} \right] \\
&\geq \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right] \quad \text{(Jensen's Inequality)}
\end{align}
$$

Equivalently, with chain rule:

$$
\begin{align}
\log p(x) &= \mathbb{E}_{q_\phi(z \mid x)}[\log p(x)] \\
&= \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{p(z \mid x)} \right] \\
&= \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z) q_\phi(z \mid x)}{p(z \mid x) q_\phi(z \mid x)} \right] \\
&= \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right] + D_{\text{KL}}(q_\phi(z \mid x) \| p(z \mid x))
\end{align}
$$

Since KL divergence $\geq 0$, the ELBO is indeed a lower bound. Maximizing ELBO minimizes the KL between approximate and true posterior.

**VAE setup.** Split the ELBO:

$$
\mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right] = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} \right]
= \underbrace{\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \| p(z))}_{\text{prior matching}}
$$

Encoder $q_\phi(z \mid x)$ maps $x \to$ latent distribution. Decoder $p_\theta(x \mid z)$ reconstructs $x$ from $z$. Gaussian choices:

$$
q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) \mathbf{I}), \quad p(z) = \mathcal{N}(0, \mathbf{I})
$$

**Reparameterization trick.** To backprop through stochastic $z$, write $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. Then KL term is analytic, reconstruction term uses Monte Carlo:

$$
\max_{\phi,\theta} \sum_{l=1}^L \log p_\theta(x \mid z^{(l)}) - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))
$$

### Hierarchical VAE (Markovian)

Stack $T$ latents in Markov chain. Joint and posterior:

$$
p(x, z_{1:T}) = p(z_T) p_\theta(x \mid z_1) \prod_{t=2}^T p_\theta(z_{t-1} \mid z_t)
$$

$$
q_\phi(z_{1:T} \mid x) = q_\phi(z_1 \mid x) \prod_{t=2}^T q_\phi(z_t \mid z_{t-1})
$$

ELBO becomes:

$$
\mathbb{E}_{q_\phi(z_{1:T} \mid x)} \left[ \log \frac{p(x, z_{1:T})}{q_\phi(z_{1:T} \mid x)} \right] = \mathbb{E}_{q_\phi} \left[ \log \frac{p(z_T) p_\theta(x \mid z_1) \prod_{t=2}^T p_\theta(z_{t-1} \mid z_t)}{q_\phi(z_1 \mid x) \prod_{t=2}^T q_\phi(z_t \mid z_{t-1})} \right]
$$

---

## Forward Diffusion Process

Let $x_0 \sim q(x_0)$ be data. Define a Markov chain that progressively adds Gaussian noise over $T$ steps:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

where $\{\beta_t\}_{t=1}^T$ is a variance schedule with $\beta_t \in (0,1)$.

**Reparameterization:** Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Then:

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0,I)
$$

**Closed form at arbitrary $t$:**

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)
$$

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$

As $T \to \infty$ and schedule chosen properly, $q(x_T) \approx \mathcal{N}(0,I)$.

## Reverse Process

Goal: learn $p_\theta(x_{t-1} \mid x_t)$ to reverse the diffusion. Parameterize as Markov chain:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
$$

where

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**Key insight:** For small $\beta_t$, $q(x_{t-1} \mid x_t, x_0)$ is tractable Gaussian.

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

where

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

## Training Objective (ELBO)

Maximize log-likelihood via variational lower bound:

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] = -L
$$

Decompose:

$$
L = \mathbb{E}_q \left[ \underbrace{D_{KL}(q(x_T \mid x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0 \mid x_1)}_{L_0} \right]
$$

**Simplified loss (DDPM):** Fix $\Sigma_\theta(x_t,t) = \tilde{\beta}_t I$, predict noise $\epsilon_\theta(x_t, t)$:

$$
L_{\text{simple}} = \mathbb{E}_{t \sim \mathcal{U}(1,T), x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.

## Score Matching Connection

Define score function $\nabla_{x_t} \log q(x_t)$. Tweedie's formula gives:

$$
\mathbb{E}[x_0 \mid x_t] = \frac{x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log q(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

**Noise prediction $\leftrightarrow$ Score prediction:**

$$
\epsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t} \, s_\theta(x_t, t)
$$

where $s_\theta(x_t, t) \approx \nabla_{x_t} \log q(x_t)$.

Training diffusion models via denoising is equivalent to score matching with a weighted combination of denoising score matching objectives at multiple noise levels.

## Sampling Algorithm

**DDPM sampling:**

1. Sample $x_T \sim \mathcal{N}(0,I)$
2. For $t = T, \ldots, 1$:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

where $z \sim \mathcal{N}(0,I)$ and $\sigma_t = \tilde{\beta}_t$ or $\sigma_t = 0$ for deterministic sampling.

**Mean parameterization:**

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

---

## References

Based on "Understanding Diffusion Models: A Unified Perspective" (arXiv:2208.11970) and related work.

