---
title: "VAEs and GANs on CelebA"
date: 2026-01-08
categories: [notes]
tags: [ml, vae, gan, generative-models, latent-space]
math: true
excerpt: "Built VAEs and GANs to generate faces. Explored latent space interpolation, compared approaches."
---

Trained VAEs and DCGANs on CelebA to generate faces. Implemented the full pipeline from scratch - custom transposed convs, reparameterization trick, adversarial training. Played around with latent space interpolation and PCA to see what the models learned.

## Dataset

Training on CelebA-64. Standard face generation benchmark, 200k celebrity faces preprocessed to 64×64.

![CelebA Samples]({{ "/assets/images/Screenshot 2026-01-13 at 12.14.20 AM.png" | relative_url }})

Good diversity in lighting, poses, expressions. The variety here forces the model to learn robust features rather than memorizing specific patterns.

---

## Variational Autoencoders

### The Big Picture

Before diving into math: what are we even trying to do?

We want to learn a probability distribution $p(x)$ over images $x$. If we have $p(x)$, we can sample new images. But images are high-dimensional (64×64×3 = 12,288 dims), and modeling $p(x)$ directly is intractable.

Key insight: maybe there's a low-dimensional **latent space** $z$ that captures the essential factors of variation. Like "this person has dark hair, is smiling, wearing glasses" can be encoded in 100 numbers instead of 12,288 pixels.

So we assume:
1. There's a simple prior $p(z) = \mathcal{N}(0, \mathbf{I})$ over latent codes
2. A decoder network $p_\theta(x \mid z)$ generates images from codes
3. An encoder network $q_\phi(z \mid x)$ infers codes from images

The trick: we can't compute $p(x) = \int p_\theta(x \mid z) p(z) dz$ (intractable integral over all possible $z$). So instead we maximize a **lower bound** on $\log p(x)$ called the ELBO.

### Deriving the ELBO

Let's build this up. Start with what we want: $\log p(x)$, the log-likelihood of our data.

Introduce the posterior $p(z \mid x)$ and our approximate posterior $q_\phi(z \mid x)$:

$$
\log p(x) = \log \int p(x, z) dz = \log \int \frac{p(x, z)}{q_\phi(z \mid x)} q_\phi(z \mid x) dz
$$

This is just multiplying and dividing by $q_\phi(z \mid x)$ - a trick to introduce our encoder.

Recognize this as an expectation over $q_\phi(z \mid x)$:

$$
\log p(x) = \log \mathbb{E}_{q_\phi(z \mid x)} \left[ \frac{p(x,z)}{q_\phi(z \mid x)} \right]
$$

Apply Jensen's inequality. Since $\log$ is concave:

$$
\log \mathbb{E}[X] \geq \mathbb{E}[\log X]
$$

Therefore:

$$
\log p(x) \geq \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right] = \text{ELBO}
$$

This is the **Evidence Lower BOund**. By maximizing ELBO, we push up $\log p(x)$ from below.

Expand $p(x, z) = p_\theta(x \mid z) p(z)$ and split the expectation:

$$
\begin{align}
\text{ELBO} &= \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} \right] \\
&= \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] + \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(z)}{q_\phi(z \mid x)} \right] \\
&= \underbrace{\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction term}} - \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \| p(z))}_{\text{regularization term}}
\end{align}
$$

That's the VAE objective! Two terms:

1. **Reconstruction term**: encode $x$ to $z$ via $q_\phi$, decode back via $p_\theta$. This is like the autoencoder loss.
2. **KL term**: keep $q_\phi(z \mid x)$ close to prior $p(z) = \mathcal{N}(0, \mathbf{I})$. This regularizes the latent space.

**Final loss** (we minimize the negative ELBO):

$$
\mathcal{L}(\theta, \phi; x) = -\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] + D_{\text{KL}}(q_\phi(z \mid x) \| p(z))
$$

In practice:
- Encoder outputs $\mu_\phi(x)$ and $\log \sigma_\phi^2(x)$, defining $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) \mathbf{I})$
- We use MSE for reconstruction: $-\log p_\theta(x \mid z) \approx \|x - \hat{x}\|^2$
- KL has closed form for Gaussians: $D_{\text{KL}}(q \| p) = \frac{1}{2} \sum_{i=1}^d (1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2)$

### Reparameterization Trick

Problem: we need to sample $z \sim q_\phi(z \mid x) = \mathcal{N}(\mu, \sigma^2 \mathbf{I})$ to compute the expectation, but sampling is non-differentiable!

Solution: **reparameterize** the sampling operation.

Instead of:
$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

Write:
$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

Now the randomness is in $\epsilon$, which doesn't depend on $\mu$ or $\sigma$. Gradients can flow through $\mu$ and $\sigma$ during backprop!

**Why this works:** Both formulations give the same distribution for $z$, but the second makes the dependence on $\mu, \sigma$ explicit and differentiable.

### Transpose Convolutions

Before we get to architecture, need to understand **transpose convolutions** (also called deconvolutions or fractionally-strided convolutions). They're the inverse of convolutions - they **upsample** instead of downsample.

#### Regular Convolution (Downsampling)

A regular convolution with stride $s$ can be written as matrix multiplication:

$$
y = Wx
$$

where $W$ is a sparse matrix built from the kernel, $x$ is the input (flattened), and $y$ is the output (flattened).

For example: 4×4 input → 2×2 output (stride 2, kernel 3×3). Each output pixel is a weighted sum of a 3×3 region in the input. The matrix $W$ has shape $(4, 16)$ where each row corresponds to one output pixel and has 9 non-zero entries (the kernel weights).

#### Transpose Convolution (Upsampling)

A transpose convolution applies the **transpose** of that matrix:

$$
y = W^T x
$$

where $x$ is now the small input and $y$ is the larger output.

**What does this mean geometrically?**

Instead of combining multiple input pixels into one output pixel, we're doing the reverse: spreading one input pixel across multiple output pixels!

Here's how it works:
- Take input pixel at position $(i, j)$
- Multiply it by the entire kernel to get a patch
- Place this patch in the output at position $(s \cdot i, s \cdot j)$ where $s$ is stride
- Where patches from different inputs overlap, **sum** them

**Concrete example:** 2×2 input → 4×4 output (stride 2, kernel 3×3, padding 1)

```
Input:           Kernel:          Output:
[a b]           [k₁₁ k₁₂ k₁₃]     [output is 4×4]
[c d]           [k₂₁ k₂₂ k₂₃]
                [k₃₁ k₃₂ k₃₃]
```

Each input value spreads a 3×3 pattern:
- Input `a` at (0,0) → kernel pattern placed at output (0,0)
- Input `b` at (0,1) → kernel pattern placed at output (0,2) [stride 2!]
- Input `c` at (1,0) → kernel pattern placed at output (2,0)
- Input `d` at (1,1) → kernel pattern placed at output (2,2)

Where these 3×3 patches overlap in the output, we add the values together.

**Why use kernel size 4 for stride 2?**

Rule of thumb: $\text{kernel\_size} = 2 \times \text{stride}$ gives clean upsampling with predictable output size and minimal checkerboard artifacts.

Output size formula:
$$
H_{\text{out}} = (H_{\text{in}} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size}
$$

For stride=2, kernel=4, padding=1:
$$
H_{\text{out}} = (H_{\text{in}} - 1) \times 2 - 2 + 4 = 2H_{\text{in}}
$$

Exactly 2× upsampling!

### Architecture

Encoder: 4 conv blocks (stride 2, BatchNorm, ReLU) → flatten → split to $\mu$, $\log \sigma^2$.

Decoder: Linear projection → 4 **transposed conv** blocks (kernel 4, stride 2, padding 1) → tanh output.

Latent dim 100, hidden channels $[128, 256, 512, 1024]$.

```python
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            t.randn(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(t.zeros(out_channels))

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, self.bias,
                                 stride=self.stride, padding=self.padding)


class VAE(nn.Module):
    def __init__(self, latent_dim_size=100, hidden_dim_size=128):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim_size),
            nn.ReLU(),
            nn.Conv2d(hidden_dim_size, hidden_dim_size * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim_size * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim_size * 2, hidden_dim_size * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim_size * 4),
            nn.ReLU(),
            nn.Conv2d(hidden_dim_size * 4, hidden_dim_size * 8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim_size * 8),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.flatten_size = hidden_dim_size * 8 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim_size)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim_size)

        self.decoder_input = nn.Linear(latent_dim_size, self.flatten_size)
        self.decoder = nn.Sequential(
            Rearrange("b (c h w) -> b c h w", c=hidden_dim_size * 8, h=4, w=4),
            ConvTranspose2d(hidden_dim_size * 8, hidden_dim_size * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim_size * 4),
            nn.ReLU(),
            ConvTranspose2d(hidden_dim_size * 4, hidden_dim_size * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim_size * 2),
            nn.ReLU(),
            ConvTranspose2d(hidden_dim_size * 2, hidden_dim_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim_size),
            nn.ReLU(),
            ConvTranspose2d(hidden_dim_size, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = t.exp(0.5 * logvar)
        epsilon = t.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x, x_recon, mu, logvar, beta_kl=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * t.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta_kl * kl_loss, recon_loss, kl_loss
```

### Training & Results

Trained for 100 epochs with batch size 64, Adam lr=5e-4. Used $\beta_{\text{KL}} = 0.03$ to prioritize reconstruction quality - lower beta means the model focuses more on making sharp images rather than perfect latent structure.

The training was pretty smooth. Reconstruction loss dominated early on, then KL divergence kicked in to regularize the space.

![Autoencoder Latent Space]({{ "/assets/images/Screenshot 2026-01-13 at 12.14.38 AM.png" | relative_url }})

Here's what the autoencoder baseline looks like - it reconstructs well but the latent space visualization shows repeated patterns. Without the KL term, there's no pressure to organize the latent space nicely. This grid is generated by sweeping across two dimensions of the 100D latent space.

![VAE Latent Space]({{ "/assets/images/Screenshot 2026-01-13 at 12.14.47 AM.png" | relative_url }})

VAE latent space looks much more structured. The faces transition smoothly across the grid, showing the model learned a continuous representation. Each position in this grid corresponds to a different point in the first two latent dimensions.

---

## Latent Space Interpolation

### Linear Interpolation

Straight line between latent vectors:

$$
z_\alpha = (1 - \alpha) z_1 + \alpha z_2, \quad \alpha \in [0, 1]
$$

![Latent Space Interpolation]({{ "/assets/images/Screenshot 2026-01-13 at 12.14.51 AM.png" | relative_url }})

Pretty smooth! The face gradually morphs from one person to another. Hair color, skin tone, lighting all transition continuously. This works because the VAE learned to organize similar features near each other in latent space.

### Spherical Interpolation (SLERP)

**Why not just use linear interpolation?**

Linear interpolation works in Euclidean space but ignores an important fact: the VAE's prior is $\mathcal{N}(0, \mathbf{I})$, and most probability mass concentrates on a **hypersphere** of radius $\sqrt{d}$ (where $d$ is latent dimension).

Walking a straight line from $z_1$ to $z_2$ cuts through the interior of the sphere, passing through low-density regions. SLERP walks along the **great circle** on the sphere surface, staying in high-density regions.

Great circle interpolation between two points on a sphere:

$$
\text{slerp}(z_1, z_2, \alpha) = \frac{\sin((1-\alpha)\omega)}{\sin \omega} z_1 + \frac{\sin(\alpha \omega)}{\sin \omega} z_2
$$

where:
- $\omega$ is the angle between vectors: $\omega = \arccos(\hat{z}_1 \cdot \hat{z}_2)$
- $\hat{z} = z / \|z\|$ normalizes to unit sphere
- $\alpha \in [0, 1]$ is interpolation parameter

**Why this formula?**

Think of $z_1$ and $z_2$ as points on a unit sphere. We want the path that:
1. Stays on the sphere surface (maintains constant distance from origin)
2. Has constant angular velocity (uniform speed along arc)

The weights $\frac{\sin((1-\alpha)\omega)}{\sin \omega}$ and $\frac{\sin(\alpha \omega)}{\sin \omega}$ accomplish this. They're derived from the requirement that the interpolated point stays on the sphere and moves at constant angular speed.

**Geometric intuition:**

```
Linear:  z₁ -------- z_mid -------- z₂  (cuts through interior)
                   ↓
SLERP:   z₁ ~~~~~~~~ z_mid ~~~~~~~~ z₂  (follows surface)
```

On the unit sphere, SLERP is the unique path with constant speed that connects two points along the shortest arc.

Result: smoother transitions, better samples. SLERP stays where the latent distribution has mass.

```python
def slerp(z1, z2, alpha):
    z1_norm = z1 / z1.norm(dim=-1, keepdim=True)
    z2_norm = z2 / z2.norm(dim=-1, keepdim=True)

    omega = t.acos((z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1))
    sin_omega = t.sin(omega)

    if (sin_omega.abs() < 1e-6).any():
        return (1 - alpha) * z1 + alpha * z2

    return (t.sin((1 - alpha) * omega) / sin_omega) * z1 + \
           (t.sin(alpha * omega) / sin_omega) * z2


@t.inference_mode()
def interpolate_slerp(model, dataset, n_steps=10):
    model.eval()
    idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
    img1 = dataset[idx1][0].unsqueeze(0).to(device)
    img2 = dataset[idx2][0].unsqueeze(0).to(device)

    z1, z2 = model.encode(img1)[0], model.encode(img2)[0]

    alphas = t.linspace(0, 1, n_steps).unsqueeze(1).to(device)
    images = []
    for alpha in alphas:
        z = slerp(z1, z2, alpha)
        img = model.decode(z)
        images.append(img)

    return t.cat(images)
```

---

## Random Sampling

Sample from prior $\mathcal{N}(0, \mathbf{I})$:

![Random Samples from Prior]({{ "/assets/images/Screenshot 2026-01-13 at 12.14.54 AM.png" | relative_url }})

These faces are completely generated from random noise - not reconstructions. The KL term forced the encoder to match the prior distribution, so sampling from $\mathcal{N}(0, \mathbf{I})$ actually produces valid faces. You can see good diversity in gender, age, lighting, and expression.

---

## PCA Analysis

### What is PCA?

**Principal Component Analysis** finds the directions of maximum variance in high-dimensional data.

Given $N$ data points $\{z_1, z_2, \ldots, z_N\}$ where each $z_i \in \mathbb{R}^d$ (in our case, $d=100$ latent dimensions), we want to find a lower-dimensional representation that captures most of the variation.

Center the data:

$$
\bar{z} = \frac{1}{N} \sum_{i=1}^N z_i
$$

$$
\tilde{z}_i = z_i - \bar{z}
$$

Compute covariance matrix:

$$
C = \frac{1}{N} \sum_{i=1}^N \tilde{z}_i \tilde{z}_i^T \in \mathbb{R}^{d \times d}
$$

This matrix encodes how each dimension varies with every other dimension. $C_{ij}$ measures correlation between dimensions $i$ and $j$.

Find eigenvectors of $C$:

$$
C v_k = \lambda_k v_k
$$

The eigenvectors $v_k$ are the **principal components** - orthogonal directions in latent space. The eigenvalues $\lambda_k$ measure how much variance is in that direction.

Sort by eigenvalue: $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d$. Then:
- $v_1$ is the direction of maximum variance (first principal component)
- $v_2$ is the direction of maximum variance orthogonal to $v_1$ (second PC)
- etc.

To represent data point $z$ in the PC basis:

$$
z_{\text{PC}} = V^T (z - \bar{z})
$$

where $V = [v_1 \; v_2 \; \ldots \; v_k]$ contains the top $k$ principal components.

To go back to original space:

$$
z_{\text{recon}} = V z_{\text{PC}} + \bar{z}
$$

If we use all $d$ components, reconstruction is perfect. If we use only top $k$ components, we keep the $k$ dimensions with most variance and lose the rest.

The fraction of total variance captured by top $k$ components is:

$$
\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}
$$

### Applying PCA to VAE Latents

Encoded 5000 images, ran PCA on the 100-dimensional latents. Top 2 components capture the main variance axes.

![PCA Latent Space Grid]({{ "/assets/images/Screenshot 2026-01-13 at 12.15.00 AM.png" | relative_url }})

Generated a 10×10 grid by sweeping across the first two principal components. Horizontal axis is PC1, vertical is PC2. The smooth transitions show the VAE learned a relatively linear structure - features combine additively rather than in complex nonlinear ways.

What's interesting here:
- Most variance concentrates in just a few dimensions. The top 10-20 PCs probably capture 80%+ of variation.
- Principal components often correspond to semantic features. Walking along PC1 might gradually change lighting or pose, PC2 might control gender or age.
- We used 100 latent dims but could probably get similar results with 20-30 dims. The KL term pushed the encoder to use fewer effective dimensions.

```python
@t.inference_mode()
def pca_analysis(model, dataset, n_samples=5000):
    model.eval()

    n_samples = min(n_samples, len(dataset))
    latents = []

    for i in range(0, n_samples, 100):
        batch_size = min(100, n_samples - i)
        imgs = t.stack([dataset[j][0] for j in range(i, i + batch_size)]).to(device)
        mu, _ = model.encode(imgs)
        latents.append(mu.cpu())

    latents = t.cat(latents, dim=0).numpy()

    pca = PCA(n_components=2)
    pca.fit(latents)

    n_points = 10
    x = t.linspace(-3, 3, n_points)
    grid_2d = t.stack([
        einops.repeat(x, "d1 -> d1 d2", d2=n_points),
        einops.repeat(x, "d2 -> d1 d2", d1=n_points),
    ], dim=-1)

    pca_components = t.from_numpy(pca.components_).float().to(device)
    grid_2d = grid_2d.to(device)
    grid_latent = grid_2d @ pca_components
    grid_flat = grid_latent.view(-1, model.latent_dim_size)

    generated = []
    for i in range(0, len(grid_flat), 50):
        batch = grid_flat[i:i+50]
        imgs = model.decode(batch).cpu()
        generated.append(imgs)

    return t.cat(generated, dim=0).view(n_points, n_points, 3, 64, 64)
```

---

## Generative Adversarial Networks

### The Big Picture

VAEs learn to generate by maximizing likelihood. GANs take a completely different approach: **adversarial training**.

Setup:
1. **Generator** $G$: Takes random noise $z \sim \mathcal{N}(0, \mathbf{I})$ and produces fake images $G(z)$
2. **Discriminator** $D$: Binary classifier that tries to distinguish real images from fake ones

They play a game:
- $D$ tries to give high scores to real images, low scores to fake images
- $G$ tries to make fake images that fool $D$ into giving high scores

At equilibrium, $G$ produces images so realistic that $D$ can't tell them apart from real ones: $D(G(z)) = 0.5$ everywhere.

### The Minimax Objective

Let's construct the objective from the ground up.

**What does the discriminator want?**

For a real image $x \sim p_{\text{data}}$:
- Want $D(x) \approx 1$ (classify as real)
- Binary cross-entropy: $-\log D(x)$ is small when $D(x)$ is close to 1

For a fake image $G(z)$ where $z \sim p(z)$:
- Want $D(G(z)) \approx 0$ (classify as fake)
- Binary cross-entropy: $-\log(1 - D(G(z)))$ is small when $D(G(z))$ is close to 0

Discriminator **maximizes** expected log-probability of correct classification:

$$
\max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
$$

**What does the generator want?**

Generator wants to **minimize** that same objective (it wants $D$ to fail):

$$
\min_G \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
$$

Put them together, we get a **minimax game**:

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
$$

**Why is this a "game"?** $D$ and $G$ have opposing objectives. $D$ wants to maximize $V$, $G$ wants to minimize it. Training alternates between updating $D$ and $G$.

### Practical Training Objectives

The minimax formulation above is theoretically nice but has a practical problem: early in training, when $G$ is terrible, $D$ easily rejects fake images: $D(G(z)) \approx 0$.

Then $\log(1 - D(G(z))) \approx 0$, giving very small gradients for $G$. Training is slow!

**Solution:** Instead of minimizing $\log(1 - D(G(z)))$, maximize $\log D(G(z))$. Same optimal solution but stronger gradients early on.

Training loop alternates:
- **Update $D$**: Maximize $\log D(x) + \log(1 - D(G(z)))$ by doing a forward pass on real images, generating fake images, and doing gradient ascent on the sum.
- **Update $G$**: Maximize $\log D(G(z))$ by generating fake images, forwarding through $D$, and doing gradient ascent.

### Equilibrium Analysis

At the optimal solution, what happens?

**Claim:** At equilibrium, $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$ where $p_g$ is the distribution induced by $G$.

**Why?** For fixed $G$, optimal $D$ maximizes:

$$
V(D) = \int_x p_{\text{data}}(x) \log D(x) dx + \int_x p_g(x) \log(1 - D(x)) dx
$$

Taking derivative w.r.t. $D(x)$ and setting to zero:

$$
\frac{p_{\text{data}}(x)}{D(x)} - \frac{p_g(x)}{1 - D(x)} = 0
$$

Solving: $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$

**At perfect equilibrium:** If $G$ generates perfectly ($p_g = p_{\text{data}}$), then $D^*(x) = 0.5$ everywhere. The discriminator can't tell real from fake!

### DCGAN Architecture

Generator: Project noise to spatial, 4 transposed conv blocks, BatchNorm everywhere except output, ReLU activations, tanh output.

Discriminator: 4 conv blocks, LeakyReLU(0.2), BatchNorm after first layer, no pooling (strided convs), sigmoid output.

Weight init: Normal(0, 0.02) for conv layers, Normal(1.0, 0.02) for BatchNorm.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim_size=100, img_size=64, img_channels=3,
                 hidden_channels=[128, 256, 512]):
        super().__init__()
        n_layers = len(hidden_channels)
        hidden_channels = hidden_channels[::-1]
        self.latent_dim_size = latent_dim_size

        first_h = img_size // (2**n_layers)
        first_size = hidden_channels[0] * (first_h**2)

        self.project_and_reshape = nn.Sequential(
            nn.Linear(latent_dim_size, first_size, bias=False),
            Rearrange("b (ic h w) -> b ic h w", h=first_h, w=first_h),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(),
        )

        conv_layers = []
        for i, (c_in, c_out) in enumerate(zip(hidden_channels,
                                               hidden_channels[1:] + [img_channels])):
            layers = [ConvTranspose2d(c_in, c_out, 4, 2, 1)]
            if i < n_layers - 1:
                layers += [nn.BatchNorm2d(c_out), nn.ReLU()]
            else:
                layers.append(nn.Tanh())
            conv_layers.append(nn.Sequential(*layers))

        self.hidden_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.project_and_reshape(x)
        return self.hidden_layers(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return t.where(x > 0, x, self.negative_slope * x)


class Discriminator(nn.Module):
    def __init__(self, img_size=64, img_channels=3, hidden_channels=[128, 256, 512]):
        super().__init__()
        n_layers = len(hidden_channels)

        conv_layers = []
        for i, (c_in, c_out) in enumerate(zip([img_channels] + hidden_channels[:-1],
                                               hidden_channels)):
            layers = [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False)]
            if i > 0:
                layers.insert(1, nn.BatchNorm2d(c_out))
            layers.append(LeakyReLU(0.2))
            conv_layers.append(nn.Sequential(*layers))

        self.hidden_layers = nn.Sequential(*conv_layers)

        final_size = hidden_channels[-1] * ((img_size // (2**n_layers)) ** 2)
        self.classifier = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(final_size, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.classifier(x).squeeze()


class DCGAN(nn.Module):
    def __init__(self, latent_dim_size=100, img_size=64, img_channels=3,
                 hidden_channels=[128, 256, 512]):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.netG = Generator(latent_dim_size, img_size, img_channels, hidden_channels)
        self.netD = Discriminator(img_size, img_channels, hidden_channels)
        self.apply(initialize_weights)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (ConvTranspose2d, nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
```

### Training & Results

Trained for 200 epochs with alternating D/G updates. Adam lr=0.0002, $\beta = (0.5, 0.999)$. Added gradient clipping (norm 1.0) for stability - GANs can be finicky without it.

```python
def train_gan(gan, dataset, epochs=200, batch_size=32, lr=0.0002):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optG = t.optim.Adam(gan.netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = t.optim.Adam(gan.netD.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for img_real, _ in loader:
            img_real = img_real.to(device)
            bs = img_real.size(0)
            noise = t.randn(bs, gan.latent_dim_size).to(device)
            img_fake = gan.netG(noise)

            optD.zero_grad()
            D_x = gan.netD(img_real)
            D_G_z = gan.netD(img_fake.detach())
            lossD = -(t.log(D_x + 1e-8).mean() + t.log(1 - D_G_z + 1e-8).mean())
            lossD.backward()
            nn.utils.clip_grad_norm_(gan.netD.parameters(), 1.0)
            optD.step()

            optG.zero_grad()
            D_G_z = gan.netD(img_fake)
            lossG = -(t.log(D_G_z + 1e-8).mean())
            lossG.backward()
            nn.utils.clip_grad_norm_(gan.netG.parameters(), 1.0)
            optG.step()
```

Training stabilized nicely. The losses bounce around as the networks compete, but eventually reach an equilibrium where neither dominates.

![GAN Latent Space Interpolation]({{ "/assets/images/Screenshot 2026-01-13 at 12.15.05 AM.png" | relative_url }})

GAN interpolation is smooth despite having no explicit latent structure. The generator learned a continuous manifold naturally through adversarial training. Walking between two random points gives a believable morph - lighting, face shape, expression all transition coherently.

![GAN Generated Faces]({{ "/assets/images/Screenshot 2026-01-13 at 12.15.10 AM.png" | relative_url }})

Generated faces look noticeably sharper than VAE samples. The discriminator acts as a learned perceptual loss - it forces the generator to produce images that look realistic at a high level, not just match pixels. You can see diverse ages, genders, lighting conditions, and expressions.

```python
@t.inference_mode()
def gan_interpolate_slerp(gan, n_steps=10):
    gan.netG.eval()

    z1 = t.randn(1, gan.latent_dim_size).to(device)
    z2 = t.randn(1, gan.latent_dim_size).to(device)

    alphas = t.linspace(0, 1, n_steps).unsqueeze(1).to(device)
    images = []

    for alpha in alphas:
        z = slerp(z1, z2, alpha)
        img = gan.netG(z)
        images.append(img)

    return t.cat(images)
```

---

## Takeaways

Both VAEs and GANs learn meaningful latent representations, just with different tradeoffs.

**VAE:**
- Explicit likelihood objective makes training stable
- KL term enforces structured latent space
- Samples can be blurry due to pixel-wise MSE loss
- Built-in encoder for inference

**GAN:**
- Sharper samples thanks to learned perceptual loss
- Training less stable - need to carefully balance discriminator and generator
- No encoder by default (though you can add one)
- Can suffer mode collapse if training goes wrong

**Other observations:**

SLERP consistently beats linear interpolation. It respects the hypersphere geometry where latent density naturally concentrates.

PCA revealed that most variance lives in a small number of dimensions. We could probably compress from 100 to 20-30 dims without losing much quality.

Implementing custom ConvTranspose2d was worth it for understanding how upsampling really works under the hood.

Trained everything on CUDA with no device bugs. Saved checkpoints for potential downstream use.
