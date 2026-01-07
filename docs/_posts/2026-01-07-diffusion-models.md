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

**Getting the ELBO.** Workaround: introduce an approximate posterior $q_\phi(z \mid x)$ that we *can* work with, then derive a lower bound on log evidence.

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

**VAE breakdown.** Split ELBO into two pieces:

$$
\mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right] = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} \right]
= \underbrace{\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \| p(z))}_{\text{prior matching}}
$$

Encoder $q_\phi(z \mid x)$ compresses $x$ into distribution over latents, decoder $p_\theta(x \mid z)$ reconstructs original $x$. Standard setup uses Gaussians:

$$
q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) \mathbf{I}), \quad p(z) = \mathcal{N}(0, \mathbf{I})
$$

**Reparameterization trick.** Can't backprop through sampling, so rewrite: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. Randomness gets separated out so we can differentiate through $\mu_\phi$ and $\sigma_\phi$. KL term becomes closed-form, reconstruction gets Monte Carlo'd:

$$
\max_{\phi,\theta} \sum_{l=1}^L \log p_\theta(x \mid z^{(l)}) - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))
$$

### Hierarchical VAE (Markovian)

What if instead of one latent, we chain together $T$ of them? Joint distribution and posterior:

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

Now we're ready for diffusion. Take the HVAE and add three restrictions: (1) latent dim = data dim, (2) encoder at each timestep is fixed linear Gaussian (no learning!), (3) Gaussian params vary so $p(x_T)$ becomes standard normal.

Notation switch: $x_t$ where $t=0$ is data, $t \in [1,T]$ are latents.

**Forward process (adding noise):**

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})
$$

where transitions are fixed:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)\mathbf{I})
$$

**Reverse process (learning to denoise):**

$$
p(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t), \quad p(x_T) = \mathcal{N}(0, \mathbf{I})
$$

**ELBO derivation.** Same game as before - start from log evidence:

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

Key trick: rewrite encoder $q(x_t \mid x_{t-1})$ using Bayes to condition on $x_0$:

$$
q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0) q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}
$$

After algebra (full derivation in [Sohl-Dickstein et al.](https://arxiv.org/abs/2006.11239) and [this paper](https://arxiv.org/pdf/2208.11970)):

$$
\begin{align}
\log p(x) &\geq \mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)] \\
&\quad - D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T)) \\
&\quad - \sum_{t=2}^T \mathbb{E}_{q(x_{t-1}, x_{t+1} \mid x_0)} \left[ D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t)) \right]
\end{align}
$$

Three terms: (1) reconstruction, (2) prior matching (zero with good schedule), (3) denoising matching - learned reverse $p_\theta(x_{t-1} \mid x_t)$ needs to match ground-truth $q(x_{t-1} \mid x_t, x_0)$.

Variance is lower than standard HVAE since each expectation is over at most one random variable.

### Computing the Ground Truth Posterior

To make the denoising term tractable, need $q(x_t \mid x_0)$ and $q(x_{t-1} \mid x_t, x_0)$. Use reparameterization:

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, \mathbf{I})
$$

Apply recursively (sum of Gaussians is Gaussian):

$$
\begin{align}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}^* \\
&= \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}} \epsilon_{t-2}^*) + \sqrt{1-\alpha_t} \epsilon_{t-1}^* \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t - \alpha_t \alpha_{t-1}} \epsilon_{t-2}^* + \sqrt{1-\alpha_t} \epsilon_{t-1}^* \\
&= \cdots \\
&= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_0
\end{align}
$$

where $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$. Thus:

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

For the reverse direction $q(x_{t-1} \mid x_t, x_0)$, use Bayes (derivation in paper):

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \mu_q(x_t, x_0), \sigma_q^2(t) \mathbf{I})
$$

where

$$
\mu_q(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

$$
\sigma_q^2(t) = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
$$

### Simplifying the Objective

Variances match, so minimizing KL reduces to matching means. Set $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_q^2(t) \mathbf{I})$ and match $\mu_\theta$ to $\mu_q$.

Parameterize $\mu_\theta$ by predicting noise $\epsilon$:

$$
\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t} \hat{x}_\theta(x_t, t) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

where $\hat{x}_\theta(x_t, t)$ predicts $x_0$ from noisy $x_t$. Final objective:

$$
\min_\theta \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2 \right]
$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ and $t \sim \mathcal{U}(1, T)$.

**Sampling:** Start with $x_T \sim \mathcal{N}(0, \mathbf{I})$. For $t = T, \ldots, 1$, iteratively denoise using learned transitions.

### Learning the Noise Schedule via SNR

So far $\alpha_t$ has been fixed. But what if we learn it too? Key insight: reparameterize via **signal-to-noise ratio (SNR)**.

Recall $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$. SNR at timestep $t$:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

High SNR = mostly signal, low SNR = mostly noise.

Rewrite objective in terms of SNR. Starting from KL minimization (see paper eqs 101-108):

$$
\frac{1}{2\sigma_q^2(t)} \frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{(1-\bar{\alpha}_t)^2} \| \hat{x}_\theta(x_t, t) - x_0 \|_2^2
$$

Substitute $\sigma_q^2(t) = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$ and simplify:

$$
\begin{align}
&= \frac{1}{2} \left( \frac{\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}} - \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t} \right) \| \hat{x}_\theta(x_t, t) - x_0 \|_2^2 \\
&= \frac{1}{2} \left( \text{SNR}(t-1) - \text{SNR}(t) \right) \| \hat{x}_\theta(x_t, t) - x_0 \|_2^2
\end{align}
$$

Beautiful! Objective gets weighted by *change* in SNR. Going forward decreases SNR (more noise), so $\text{SNR}(t-1) - \text{SNR}(t)$ stays positive.

**Learning the schedule.** Parameterize:

$$
\text{SNR}(t) = \exp(-\omega_\eta(t))
$$

where $\omega_\eta(t)$ is monotonically increasing network with params $\eta$. As $t$ goes up, $\omega_\eta(t)$ goes up, so $\text{SNR}(t)$ goes down. Then:

$$
\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t} = \exp(-\omega_\eta(t))
$$

Solving for $\bar{\alpha}_t$:

$$
\bar{\alpha}_t = \text{sigmoid}(-\omega_\eta(t))
$$

$$
1 - \bar{\alpha}_t = \text{sigmoid}(\omega_\eta(t))
$$

Now jointly optimize denoising network $\theta$ and noise schedule $\eta$!

### Three Ways to Train a Diffusion Model

<details>
<summary><b>Optional: Three equivalent parameterizations</b></summary>

Turns out there are three mathematically equivalent ways to train the same model. We've been predicting clean $x_0$, but can also predict noise $\epsilon_0$ or score $\nabla_{x_t} \log p(x_t)$.

**1. Predicting noise $\epsilon_0$**

From forward process:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_0
$$

Solve for $x_0$:

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_0}{\sqrt{\bar{\alpha}_t}}
$$

Plug into $\mu_q(x_t, x_0)$ (skipping algebra):

$$
\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}} \epsilon_0
$$

Train network $\hat{\epsilon}_\theta(x_t, t)$ to predict noise:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}} \hat{\epsilon}_\theta(x_t, t)
$$

Objective:

$$
\min_\theta \mathbb{E}_{t, x_0, \epsilon_0} \left[ \| \epsilon_0 - \hat{\epsilon}_\theta(x_t, t) \|_2^2 \right]
$$

Noise prediction works better empirically!

**2. Predicting score $\nabla_{x_t} \log p(x_t)$**

Score function = direction that increases log prob. Arrow pointing "uphill" toward likely images.

**Tweedie's Formula** (empirical Bayes) connects posterior mean to score. For Gaussian $z \sim \mathcal{N}(z; \mu, \Sigma)$:

$$
\mathbb{E}[\mu \mid z] = z + \Sigma \nabla_z \log p(z)
$$

Apply to $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$:

$$
\mathbb{E}[\mu_{x_t} \mid x_t] = x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log p(x_t)
$$

True mean is $\mu_{x_t} = \sqrt{\bar{\alpha}_t} x_0$, so:

$$
\sqrt{\bar{\alpha}_t} x_0 = x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log p(x_t)
$$

$$
x_0 = \frac{x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

Plug into $\mu_q$:

$$
\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} x_t + \frac{1-\alpha_t}{\sqrt{\alpha_t}} \nabla_{x_t} \log p(x_t)
$$

Train network $s_\theta(x_t, t)$ to predict score:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} x_t + \frac{1-\alpha_t}{\sqrt{\alpha_t}} s_\theta(x_t, t)
$$

Objective:

$$
\min_\theta \mathbb{E}_{t, x_0} \left[ \| \nabla_{x_t} \log p(x_t) - s_\theta(x_t, t) \|_2^2 \right]
$$

**Connection:** Combine expressions for $x_0$:

$$
\frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_0}{\sqrt{\bar{\alpha}_t}} = \frac{x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

Rearranging:

$$
\nabla_{x_t} \log p(x_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_0
$$

Score and noise differ by scaling! Score points opposite to noise (noise corrupts, score points toward clean).

Learning to denoise = learning the score. This connects diffusion to **score-based models**.

</details>

Most implementations predict noise $\epsilon_0$ (trains more stably). But score interpretation connects to score matching/Langevin dynamics and gives intuition: model learns direction toward "realistic" images.

---

*See [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970), [Lilian Weng's blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), and [DDPM](https://arxiv.org/abs/2006.11239) for full derivations and further reading.*
