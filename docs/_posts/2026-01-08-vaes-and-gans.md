---
title: "VAEs and GANs on CelebA"
date: 2026-01-08
categories: [notes]
tags: [ml, vae, gan, generative-models, latent-space]
toc: true
toc_sticky: true
math: true
---

## Intro

Built VAEs and DCGANs from scratch on CelebA-64. Trained models to generate faces, explored latent space interpolation, analyzed what these representations capture. Both approaches work but make different tradeoffs.

What's covered:
- Custom transposed convolutions
- Standard autoencoder baseline
- Full VAE with reparameterization
- DCGAN with adversarial training
- Latent space interpolation (linear, spherical, rotation)
- PCA analysis of learned representations

All code trained on CUDA, models converged clean.

## Dataset

Training on CelebA-64. Standard face generation benchmark, 200k celebrity faces preprocessed to 64×64.

![CelebA Samples]({{ "/assets/images/CelebA dataset.png" | relative_url }})

Good diversity in lighting, poses, expressions. Forces the model to learn robust features.

---

## Variational Autoencoders

### The ELBO

VAEs maximize the evidence lower bound. From Jensen's inequality:

$$
\log p(x) = \log \mathbb{E}_{q_\phi(z \mid x)} \left[ \frac{p(x,z)}{q_\phi(z \mid x)} \right] \geq \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x,z)}{q_\phi(z \mid x)} \right]
$$

Split into reconstruction and KL:

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))
$$

Encoder $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) \mathbf{I})$ compresses to Gaussian. Decoder $p_\theta(x \mid z)$ reconstructs. Prior $p(z) = \mathcal{N}(0, \mathbf{I})$ regularizes latents.

Full derivation on the [diffusion page](/posts/diffusion-models/#background-elbo-and-vae).

### Reparameterization

Can't backprop through sampling. Reparameterize: $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. Stochasticity isolated, gradients flow through $\mu$ and $\sigma$.

### Architecture

Encoder: 4 conv blocks (stride 2, BatchNorm, ReLU) → flatten → split to $\mu$, $\log \sigma^2$.

Decoder: Linear projection → 4 transposed conv blocks → tanh output.

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

### Training

10 epochs, batch 64, Adam lr=1e-3. $\beta_{\text{KL}} = 0.5$ balances reconstruction vs regularization.

![VAE Training]({{ "/assets/images/vae training.png" | relative_url }})

Loss converges smoothly. Reconstruction term dominates early, KL kicks in later. Final loss ~0.17.

![VAE Latent Space]({{ "/assets/images/vae latent space.png" | relative_url }})

Reconstructions look clean. Model compressed 12,288 dims down to 100 and back with minimal loss.

---

## Autoencoder Baseline

Standard autoencoder (no variational component) for comparison:

![Autoencoder Latent Space]({{ "/assets/images/autoencoder latent space.png" | relative_url }})

Reconstructs well but latent space less structured. No regularization means scattered representations. This is why the KL term matters.

---

## Latent Space Interpolation

### Linear Interpolation

Straight line between latent vectors:

$$
z_\alpha = (1 - \alpha) z_1 + \alpha z_2, \quad \alpha \in [0, 1]
$$

![Latent Space Interpolation]({{ "/assets/images/latent space interpolation.png" | relative_url }})

Works but doesn't respect geometry.

### Spherical Interpolation (SLERP)

Interpolate along great circle:

$$
\text{slerp}(z_1, z_2, \alpha) = \frac{\sin((1-\alpha)\omega)}{\sin \omega} z_1 + \frac{\sin(\alpha \omega)}{\sin \omega} z_2
$$

where $\omega = \arccos(\hat{z}_1 \cdot \hat{z}_2)$ and $\hat{z} = z / \|z\|$.

![VAE Latent Space Interpolation]({{ "/assets/images/vae latent space interpolation.png" | relative_url }})

Smoother transitions. Maintains constant speed through latent space, stays on the manifold where density concentrates.

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

![Random Samples from Prior]({{ "/assets/images/random samples from prior.png" | relative_url }})

Generates novel faces. The KL term forces encoder distribution to match the prior, so sampling from $\mathcal{N}(0, \mathbf{I})$ actually works.

---

## PCA Analysis

Encoded 5000 images, ran PCA on latents. Top 2 components capture main variance axes.

![PCA Training]({{ "/assets/images/pca training.png" | relative_url }})

Generated grid across PC1/PC2:

![PCA Latent Space Grid]({{ "/assets/images/pca latent space grid.png" | relative_url }})

Principal components capture interpretable features (lighting, pose, expression). Most variance in few dimensions. Could probably use 20-30 dims instead of 100 without losing much.

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

### The Minimax Game

GAN training = two-player zero-sum game:

$$
\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
$$

Discriminator $D$ learns to distinguish real from fake. Generator $G$ learns to fool discriminator. At equilibrium, $D(G(z)) = 0.5$ everywhere.

Practical training uses separate objectives:
- Discriminator: maximize $\log D(x) + \log(1 - D(G(z)))$
- Generator: maximize $\log D(G(z))$ (equivalently minimize $-\log D(G(z))$)

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

### Training

5 epochs, alternating D/G updates. Adam lr=0.0002, $\beta = (0.5, 0.999)$. Gradient clipping (norm 1.0) for stability.

```python
def train_gan(gan, dataset, epochs=5, batch_size=32, lr=0.0002):
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

![GAN Training]({{ "/assets/images/gan training.png" | relative_url }})

Training stabilizes. Losses bounce as networks compete, reach equilibrium where neither dominates.

---

## GAN Results

### Generated Faces

![GAN Generated Faces]({{ "/assets/images/gan generated faces.png" | relative_url }})

Sharper than VAE samples. Generator has one job (fool discriminator) vs two (reconstruct + match prior). Discriminator acts as learned perceptual loss, better than pixel MSE.

### GAN Interpolation

![GAN Latent Space Interpolation]({{ "/assets/images/gan latent space interpolation.png" | relative_url }})

Smooth transitions despite no explicit latent structure. Generator learned continuous manifold naturally.

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

Both VAEs and GANs learn meaningful latent representations. Different tradeoffs.

**VAE:**
- Explicit likelihood, stable training
- Structured latent space (KL term enforces this)
- Samples can be blurry (pixel loss limitation)
- Built-in encoder

**GAN:**
- Sharper samples (learned perceptual loss)
- Training less stable, need to balance D/G
- No encoder by default
- Can suffer mode collapse

**Other notes:**

SLERP > linear for interpolation. Respects hypersphere geometry where latent density concentrates.

PCA shows most variance in few dimensions. Could compress to 20-30 dims without losing much.

Custom ConvTranspose2d worked clean. Good to implement once to understand internals.

Models trained on CUDA, no device bugs. Checkpoints saved for downstream use.
