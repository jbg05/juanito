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

After algebraic manipulations (full derivation in [Sohl-Dickstein et al.](https://arxiv.org/abs/2006.11239) and [this paper](https://arxiv.org/pdf/2208.11970)), we get:

$$
\begin{align}
\log p(x) &\geq \mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)] \\
&\quad - D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T)) \\
&\quad - \sum_{t=2}^T \mathbb{E}_{q(x_{t-1}, x_{t+1} \mid x_0)} \left[ D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t)) \right]
\end{align}
$$

Three terms: (1) reconstruction, (2) prior matching (typically zero with good schedule), (3) denoising matching - learned reverse $p_\theta(x_{t-1} \mid x_t)$ matches ground-truth $q(x_{t-1} \mid x_t, x_0)$.

Lower variance than standard HVAE: each expectation over at most one random variable at a time.

### Computing the Ground Truth Posterior

Need to derive $q(x_t \mid x_0)$ and $q(x_{t-1} \mid x_t, x_0)$ to make denoising term tractable. Using reparameterization:

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, \mathbf{I})
$$

Recursively applying this (sum of Gaussians is Gaussian):

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

For $q(x_{t-1} \mid x_t, x_0)$, use Bayes rule (derivation in paper):

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

Since variances match, minimizing $D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))$ reduces to matching means. Set $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_q^2(t) \mathbf{I})$ and match $\mu_\theta$ to $\mu_q$.

Can parameterize $\mu_\theta$ by predicting the noise $\epsilon$ that was added:

$$
\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t} \hat{x}_\theta(x_t, t) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

where $\hat{x}_\theta(x_t, t)$ predicts $x_0$ from noisy $x_t$. Equivalently, train a network $\epsilon_\theta(x_t, t)$ to predict the noise:

$$
\min_\theta \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2 \right]
$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ and $t \sim \mathcal{U}(1, T)$.

**Sampling:** Start with $x_T \sim \mathcal{N}(0, \mathbf{I})$. For $t = T, \ldots, 1$, iteratively denoise using learned transitions.

### Learning the Noise Schedule via SNR

So far we've treated $\alpha_t$ as fixed hyperparameters. But we can also learn them! The key insight is to reparameterize everything in terms of the **signal-to-noise ratio (SNR)**.

Recall from earlier that $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$. The signal-to-noise ratio at timestep $t$ is:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

This ratio tells us how much signal vs. noise is present at timestep $t$. High SNR means mostly signal, low SNR means mostly noise.

Now here's the magic: we can rewrite our objective in terms of SNR. Starting from the KL minimization objective (derived from equations 101-108 in paper):

$$
\frac{1}{2\sigma_q^2(t)} \frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{(1-\bar{\alpha}_t)^2} \| \hat{x}_\theta(x_t, t) - x_0 \|_2^2
$$

Substituting our variance formula $\sigma_q^2(t) = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$ and simplifying (see equations 102-108):

$$
\begin{align}
&= \frac{1}{2} \left( \frac{\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}} - \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t} \right) \| \hat{x}_\theta(x_t, t) - x_0 \|_2^2 \\
&= \frac{1}{2} \left( \text{SNR}(t-1) - \text{SNR}(t) \right) \| \hat{x}_\theta(x_t, t) - x_0 \|_2^2
\end{align}
$$

This is beautiful! The objective at each timestep is weighted by the *change* in SNR between consecutive steps. As we go forward in the diffusion process, SNR decreases (more noise), so $\text{SNR}(t-1) - \text{SNR}(t)$ is positive.

**Learning the schedule.** Instead of fixing $\alpha_t$ values, we can parameterize:

$$
\text{SNR}(t) = \exp(-\omega_\eta(t))
$$

where $\omega_\eta(t)$ is a monotonically increasing neural network with parameters $\eta$. As $t$ increases, $\omega_\eta(t)$ increases, so $\text{SNR}(t)$ decreases (more noise over time). This gives us:

$$
\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t} = \exp(-\omega_\eta(t))
$$

Solving for $\bar{\alpha}_t$:

$$
\bar{\alpha}_t = \text{sigmoid}(-\omega_\eta(t))
$$

And similarly:

$$
1 - \bar{\alpha}_t = \text{sigmoid}(\omega_\eta(t))
$$

Now we jointly optimize both the denoising network $\theta$ and the noise schedule $\eta$! The network $\omega_\eta(t)$ learns the optimal rate at which to add noise at each timestep.

### Three Equivalent Interpretations

So here's something cool: turns out there are three totally different ways to think about what a VDM is learning, and they're all mathematically equivalent! We've been training a network to predict the original clean image $x_0$ from a noisy version $x_t$. But that same network can be reframed as:

1. **Predicting the original image** $x_0$ (what we've been doing)
2. **Predicting the noise** $\epsilon_0$ that was added to create $x_t$
3. **Predicting the score function** $\nabla_{x_t} \log p(x_t)$ (gradient of log density)

All three give you the same model, just different perspectives! Let me break down why this matters.

<details>
<summary><b>Click to expand: Full mathematical derivation</b></summary>

**Interpretation 1: Predicting the noise $\epsilon_0$**

Start by rearranging how we express $x_0$ in terms of $x_t$. Remember from our forward process that:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_0
$$

Solving for $x_0$:

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_0}{\sqrt{\bar{\alpha}_t}}
$$

Now plug this into our ground-truth denoising mean $\mu_q(x_t, x_0)$ from before. After substitution and simplification (skipping the algebra), we get:

$$
\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}} \epsilon_0
$$

So instead of predicting $x_0$ directly, we can train a network $\hat{\epsilon}_\theta(x_t, t)$ to predict the noise:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}} \hat{\epsilon}_\theta(x_t, t)
$$

The optimization becomes:

$$
\min_\theta \mathbb{E}_{t, x_0, \epsilon_0} \left[ \| \epsilon_0 - \hat{\epsilon}_\theta(x_t, t) \|_2^2 \right]
$$

This noise prediction formulation often works better empirically! The network learns "what noise was added" rather than "what the clean image looks like."

**Interpretation 2: Predicting the score function $\nabla_{x_t} \log p(x_t)$**

Here's where it gets really interesting. The **score function** $\nabla_{x_t} \log p(x_t)$ tells us which direction in data space increases the log probability most. Think of it as an arrow pointing "uphill" toward more likely images.

There's a classic result called **Tweedie's Formula** (used a lot in empirical Bayes) that connects the mean of a posterior to the score. For a Gaussian $z \sim \mathcal{N}(z; \mu, \Sigma)$, Tweedie says:

$$
\mathbb{E}[\mu \mid z] = z + \Sigma \nabla_z \log p(z)
$$

Applying this to our forward process $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$, we get:

$$
\mathbb{E}[\mu_{x_t} \mid x_t] = x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log p(x_t)
$$

where the true mean is $\mu_{x_t} = \sqrt{\bar{\alpha}_t} x_0$. Rearranging:

$$
\sqrt{\bar{\alpha}_t} x_0 = x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log p(x_t)
$$

$$
x_0 = \frac{x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

Plug this back into $\mu_q$ and simplify:

$$
\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} x_t + \frac{1-\alpha_t}{\sqrt{\alpha_t}} \nabla_{x_t} \log p(x_t)
$$

So we can also train a network $s_\theta(x_t, t)$ to predict the score:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} x_t + \frac{1-\alpha_t}{\sqrt{\alpha_t}} s_\theta(x_t, t)
$$

Objective becomes:

$$
\min_\theta \mathbb{E}_{t, x_0} \left[ \| \nabla_{x_t} \log p(x_t) - s_\theta(x_t, t) \|_2^2 \right]
$$

**Connection between noise and score**

Notice something beautiful: combining our expressions for $x_0$ from interpretations 1 and 2:

$$
\frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_0}{\sqrt{\bar{\alpha}_t}} = \frac{x_t + (1-\bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

Rearranging:

$$
\nabla_{x_t} \log p(x_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_0
$$

The score function and source noise differ only by a constant scaling factor! The score points in the opposite direction of the noise (makes sense - noise corrupts images, score points toward cleaner ones).

This means learning to denoise is *exactly the same* as learning the score function. This connection to **score-based models** is why diffusion models are so powerful.

</details>

**Why this matters:** In practice, most implementations predict the noise $\epsilon_0$ since it tends to train more stably. But understanding the score interpretation connects diffusion models to a whole other framework (score matching, Langevin dynamics) and gives us intuition: the model learns which direction to move in image space to make images more "realistic."

---

*See [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970), [Lilian Weng's blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), and [DDPM](https://arxiv.org/abs/2006.11239) for full derivations and further reading.*

