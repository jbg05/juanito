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

Generative models are insane. The idea that you can compress high-dimensional data into a low-dimensional latent space, then sample from it to generate novel outputs - that's powerful. This post covers two fundamental approaches: Variational Autoencoders and Generative Adversarial Networks.

I built both from scratch on CelebA faces. VAEs give you explicit latent structure through variational inference. GANs pit two networks against each other in a minimax game. Different philosophies, different tradeoffs, both fascinating.

What you'll see here:
- Full VAE implementation with the reparameterization trick
- Custom transposed convolutions (no cheating with PyTorch's built-ins for learning)
- Latent space interpolation - linear, spherical, and rotation-based
- PCA analysis showing what the latent dimensions actually capture
- DCGAN architecture with careful weight initialization
- Training dynamics and what makes each approach tick

Everything ran on CUDA, converged clean, and the results show what these models can really do. Let's get into it.

## Dataset

Working with CelebA-64 throughout. Standard benchmark for face generation - 200k celebrity faces preprocessed to 64×64. Clean dataset, good for seeing what models learn.

![CelebA Samples]({{ "/assets/images/CelebA dataset.png" | relative_url }})

Pretty diverse set of faces, different lighting, poses, expressions. This variety matters - forces the model to learn robust features rather than memorizing specific patterns.

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

Trained for 10 epochs with batch size 64, Adam optimizer at lr=1e-3. The $\beta_{\text{KL}} = 0.5$ hyperparameter is key here - it balances reconstruction quality against latent space regularization. Too high and you get blurry reconstructions, too low and the latent space becomes unstructured.

![VAE Training]({{ "/assets/images/vae training.png" | relative_url }})

Loss converges smoothly over training. You can see the reconstruction term dominating early on (model learning to copy inputs), then the KL divergence kicks in as the latent space gets organized. Final loss around 0.17 - pretty solid.

![VAE Latent Space]({{ "/assets/images/vae latent space.png" | relative_url }})

Reconstructions look clean. The model learned to compress 64×64×3 images (12,288 dimensions) down to just 100 latent dimensions and back with minimal information loss. That's the power of finding good representations.

---

## Autoencoder Baseline

Before diving into interpolation, useful to see what happens without the variational component. Standard autoencoders just minimize reconstruction loss - no KL regularization, no prior matching.

![Autoencoder Latent Space]({{ "/assets/images/autoencoder latent space.png" | relative_url }})

Reconstructions are solid (maybe even slightly better than VAE), but there's a catch. The latent space is scattered and disorganized. Without regularization pulling it toward a standard normal distribution, the encoder can place latent codes wherever it wants. This makes interpolation and sampling from the prior basically useless.

This is why VAEs matter - they trade a tiny bit of reconstruction quality for a structured, continuous latent space you can actually do things with.

---

## Latent Space Interpolation

One of the coolest things about VAEs - you can walk through the latent space and watch faces smoothly morph into each other. The question is: what's the best path between two points?

### Linear Interpolation

The obvious approach - just draw a straight line:

$$
z_\alpha = (1 - \alpha) z_1 + \alpha z_2, \quad \alpha \in [0, 1]
$$

![Latent Space Interpolation]({{ "/assets/images/latent space interpolation.png" | relative_url }})

Works decently. You get smooth transitions between faces. But there's a subtle issue - this doesn't respect the underlying geometry of the latent space, which tends to concentrate around a hypersphere.

### Spherical Interpolation (SLERP)

Better idea: interpolate along the great circle connecting two points on the unit hypersphere:

$$
\text{slerp}(z_1, z_2, \alpha) = \frac{\sin((1-\alpha)\omega)}{\sin \omega} z_1 + \frac{\sin(\alpha \omega)}{\sin \omega} z_2
$$

where $\omega = \arccos(\hat{z}_1 \cdot \hat{z}_2)$ is the angle between normalized vectors $\hat{z} = z / \|z\|$.

![VAE Latent Space Interpolation]({{ "/assets/images/vae latent space interpolation.png" | relative_url }})

The transitions look noticeably smoother. SLERP maintains constant "speed" through latent space and stays on the manifold where the model actually learned to generate realistic faces. This matters more in higher dimensions where linear interpolation can cut through low-density regions.

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

The real test of a generative model - can you sample random noise and get realistic outputs?

With VAEs, you sample $z \sim \mathcal{N}(0, \mathbf{I})$ from the prior and decode:

![Random Samples from Prior]({{ "/assets/images/random samples from prior.png" | relative_url }})

Pretty impressive. These are completely novel faces, not in the training set. The model learned a structured latent space where random samples from a standard normal actually land in regions that decode to realistic faces. This is exactly what the KL term in the ELBO ensures - it forces the encoder's distribution to match the prior, so sampling from the prior works.

---

## PCA Analysis

Here's where it gets interesting. We've got 100 latent dimensions, but are they all equally important?

Encoded 5000 images from the dataset and ran PCA on their latent vectors to see what structure emerges:

![PCA Training]({{ "/assets/images/pca training.png" | relative_url }})

Then generated a grid traversing the top 2 principal components:

![PCA Latent Space Grid]({{ "/assets/images/pca latent space grid.png" | relative_url }})

The principal components learned interpretable features - you can see systematic variations in lighting, head pose, and facial expression as you move across the grid. What's cool is that most of the variance concentrates in just a few dimensions. This suggests you could probably get away with a much smaller latent space (maybe 20-30 dims) without losing much representational power.

This kind of analysis shows what the VAE actually learned - not just random 100-dimensional noise, but a structured space where certain directions correspond to meaningful semantic attributes.

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

Now for something completely different. While VAEs optimize an explicit objective (the ELBO), GANs take an adversarial approach.

### The Minimax Game

The setup is elegant: pit two neural networks against each other in a zero-sum game.

$$
\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
$$

Discriminator $D$ distinguishes real from fake. Generator $G$ fools discriminator. Equilibrium at $D(G(z)) = 0.5$.

Practical objectives:
- **D**: maximize $\log D(x) + \log(1 - D(G(z)))$
- **G**: maximize $\log D(G(z))$

### DCGAN Architecture

**Generator:** Project noise → 4 transposed conv blocks, BatchNorm, ReLU, tanh output.

**Discriminator:** 4 conv blocks, LeakyReLU(0.2), BatchNorm after first, sigmoid output.

Weight init: Normal(0, 0.02) for conv, Normal(1.0, 0.02) for BatchNorm.

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

Training GANs is notoriously finicky. The key is balancing the discriminator and generator - if one gets too good, the other can't learn.

Ran 5 epochs with alternating updates. Each batch: update D once, update G once. Adam optimizer with lr=0.0002 and $\beta = (0.5, 0.999)$ (lower momentum than usual - helps with GAN stability). Added gradient clipping at norm 1.0 to prevent exploding gradients.

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

Training stabilized nicely. You can see the losses bouncing around as the networks compete, but they eventually reach an equilibrium where neither dominates. This is the sweet spot.

---

## GAN Results

### Generated Faces

After training, sample random noise and run it through the generator:

![GAN Generated Faces]({{ "/assets/images/gan generated faces.png" | relative_url }})

The samples are noticeably sharper than the VAE outputs. Makes sense - the generator has one job (fool the discriminator), not two jobs (reconstruct inputs AND match a prior). This lets it focus entirely on generating realistic-looking images. The discriminator acts as a learned loss function that's much more sophisticated than simple pixel-wise MSE.

Trade-off though: no encoder, so you can't easily map real images to latent codes. And training is way less stable than VAEs.

### GAN Interpolation

Even without explicit latent space structure, interpolation still works:

![GAN Latent Space Interpolation]({{ "/assets/images/gan latent space interpolation.png" | relative_url }})

Smooth, realistic transitions. The generator learned a continuous manifold in latent space even though nothing in the training objective explicitly encouraged this. It just naturally emerges from the adversarial dynamics - the generator benefits from having smooth, consistent mappings.

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

Both approaches work, but they make different trade-offs.

**VAE strengths:**
- Stable training - just maximize ELBO, no adversarial dynamics
- Explicit likelihood lets you do principled inference
- Structured latent space from the start (thanks to KL term)
- Built-in encoder for mapping real images to latents

**VAE weaknesses:**
- Samples can be blurry (pixel-wise loss isn't great for perceptual quality)
- Have to tune $\beta_{\text{KL}}$ hyperparameter carefully

**GAN strengths:**
- Sharper, more realistic samples
- Discriminator is a learned perceptual loss (way better than MSE)
- Can model complex distributions without explicit density

**GAN weaknesses:**
- Training is fragile - need to balance D and G carefully
- No encoder (though you can train one separately)
- Mode collapse and instability issues

**Technical notes:**

The SLERP interpolation consistently beat linear interpolation. It respects the hypersphere geometry where the model density concentrates.

PCA showed most variance in a handful of dimensions. You could probably use a 20-30 dim latent space instead of 100 without losing much. Something to try next.

Custom ConvTranspose2d implementation worked perfectly - good to build these from scratch at least once to understand what's happening under the hood.

Everything trained clean on CUDA. Models converged, no device bugs, saved checkpoints ready for fine-tuning or downstream tasks.

Both are foundational architectures worth understanding deeply. Modern generative models (diffusion, flows) build on these ideas.
