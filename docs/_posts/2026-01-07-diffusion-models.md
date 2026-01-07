---
title: "Diffusion Models: Mathematical Foundations"
date: 2026-01-07
categories: [notes]
tags: [ml, diffusion, generative-models, score-matching]
toc: true
toc_sticky: true
math: true
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

