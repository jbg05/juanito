---
title: "Transformers from Scratch"
date: 2026-01-12
categories: [notes]
tags: [ml, transformer, attention, gpt, nlp]
math: true
excerpt: "Built GPT-2 from scratch. Math-heavy walkthrough of attention, embeddings, sampling strategies."
---

Implemented the full transformer architecture - attention mechanism, positional embeddings, residual connections. Trained it to generate text and explored different sampling methods.

## The Big Picture

Transformers model sequences by learning patterns in data. For language: given tokens $x_1, \ldots, x_t$, predict distribution over next token $x_{t+1}$.

Key architectural choices:
1. **Residual stream**: Information flows straight through the model with additions, no bottlenecks
2. **Attention**: Each position looks at previous positions to gather context
3. **Causal masking**: Position $i$ cannot see positions $j > i$ (autoregressive property)
4. **Layer norm + residual**: Stabilizes deep networks

Standard setup:
- Input: token sequence $[x_1, x_2, \ldots, x_n]$
- Output: logits $\in \mathbb{R}^{n \times d_{\text{vocab}}}$, apply softmax to get probabilities
- Generation: sample from distribution, append to sequence, repeat

---

## Tokenization

Text → integers via learned sub-word units. GPT-2 uses BPE (Byte Pair Encoding) with vocabulary size 50,257.

**Why sub-words?**
- Word-level: huge vocabulary (100k+), many OOV issues
- Character-level: very long sequences, harder to learn patterns
- Sub-word: balance between vocabulary size and sequence length

**Example:**
```
"Transformers are" → [8291, 364, 389]
```

Each token maps to embedding vector via lookup table.

---

## Embeddings

### Token Embeddings

Lookup table $W_E \in \mathbb{R}^{d_{\text{vocab}} \times d_{\text{model}}}$. For token $t$, embedding is $W_E[t] \in \mathbb{R}^{d_{\text{model}}}$.

For GPT-2: $d_{\text{vocab}} = 50257$, $d_{\text{model}} = 768$.

### Positional Embeddings

Absolute position encoding via learned lookup table $W_P \in \mathbb{R}^{n_{\text{ctx}} \times d_{\text{model}}}$.

Position $i$ gets embedding $W_P[i]$. Context length $n_{\text{ctx}} = 1024$ for GPT-2.

**Combined embedding:**

$$
e_i = W_E[\text{token}_i] + W_P[i]
$$

This initialization feeds into the residual stream.

---

## Attention Mechanism

### Single-Head Attention

Core operation: for each position, compute weighted average of all previous positions' values.

**Step 1: Compute Q, K, V**

Given input $x \in \mathbb{R}^{n \times d_{\text{model}}}$:

$$
Q = x W_Q, \quad K = x W_K, \quad V = x W_V
$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$.

For GPT-2: $d_{\text{head}} = 64$.

**Step 2: Attention scores**

$$
S = \frac{QK^T}{\sqrt{d_{\text{head}}}}
$$

Scaling by $\sqrt{d_{\text{head}}}$ prevents scores from growing large (which would make softmax saturate).

**Why this scaling?** If $Q, K$ have zero mean and unit variance, then $QK^T$ has variance $d_{\text{head}}$. Dividing by $\sqrt{d_{\text{head}}}$ normalizes to unit variance.

**Step 3: Causal masking**

Set $S_{ij} = -\infty$ for $j > i$ (future positions). After softmax, these become 0.

$$
S_{\text{masked}}[i, j] = \begin{cases}
S[i, j] & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

**Step 4: Attention pattern**

$$
A = \text{softmax}(S_{\text{masked}}, \text{dim}=-1) \in \mathbb{R}^{n \times n}
$$

Row $i$ of $A$ is a probability distribution over positions $1, \ldots, i$.

**Step 5: Apply to values**

$$
\text{out} = A V \in \mathbb{R}^{n \times d_{\text{head}}}
$$

This is a weighted average of value vectors, where weights come from attention pattern.

**Geometric interpretation:**

Query at position $i$ "asks a question". Keys at positions $1, \ldots, i$ "answer" by computing similarity scores. Softmax converts scores to probabilities. Values are "information to retrieve", weighted by these probabilities.

### Multi-Head Attention

Run $h$ attention heads in parallel, concatenate outputs, project back to $d_{\text{model}}$.

$$
\text{head}_k = \text{Attention}(x W_Q^{(k)}, x W_K^{(k)}, x W_V^{(k)})
$$

$$
\text{MultiHead}(x) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

where $W_O \in \mathbb{R}^{h \cdot d_{\text{head}} \times d_{\text{model}}}$.

For GPT-2: $h = 12$ heads, $d_{\text{head}} = 64$, so $h \cdot d_{\text{head}} = 768 = d_{\text{model}}$.

**Why multi-head?**

Different heads can attend to different patterns:
- Head 1: previous token (local context)
- Head 2: nouns in sentence (syntactic patterns)
- Head 3: subject of sentence (long-range dependencies)

Each head learns different features independently, then combines via $W_O$.

---

## MLP Block

Two-layer feedforward network with GELU activation:

$$
\text{MLP}(x) = \text{GELU}(x W_{\text{in}} + b_{\text{in}}) W_{\text{out}} + b_{\text{out}}
$$

where:
- $W_{\text{in}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{mlp}}}$
- $W_{\text{out}} \in \mathbb{R}^{d_{\text{mlp}} \times d_{\text{model}}}$
- $d_{\text{mlp}} = 4 \cdot d_{\text{model}} = 3072$ for GPT-2

**GELU activation:**

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $\Phi(x) = \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$ is the Gaussian CDF.

Smooth approximation: $\text{GELU}(x) \approx x \cdot \sigma(1.702x)$ where $\sigma$ is sigmoid.

**Why GELU instead of ReLU?**

GELU is smooth (differentiable everywhere), has non-zero gradient for negative values (prevents dead neurons), empirically works better for transformers.

---

## Layer Normalization

Normalize activations to zero mean, unit variance. Apply learned scale and shift.

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:
- $\mu = \frac{1}{d} \sum_{i=1}^d x_i$ (mean over feature dimension)
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$ (variance over feature dimension)
- $\gamma, \beta \in \mathbb{R}^d$ are learned parameters
- $\epsilon = 10^{-5}$ for numerical stability

**Where it's applied in GPT-2:**

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

This is "pre-norm" architecture (normalize before the layer). Alternative is "post-norm" (normalize after).

**Why LayerNorm?**

Stabilizes training by keeping activation magnitudes consistent. Without it, deep networks suffer from vanishing/exploding gradients.

**Why not BatchNorm?**

BatchNorm normalizes across the batch dimension, depends on batch statistics. LayerNorm normalizes across features, works with any batch size (including 1).

---

## Residual Stream

Core architectural principle: information flows straight through the network via additions.

**Transformer block:**

$$
x' = x + \text{Attention}(\text{LayerNorm}(x))
$$

$$
x'' = x' + \text{MLP}(\text{LayerNorm}(x'))
$$

**Why residuals?**

1. **Gradient flow**: Backprop has direct path to input (avoids vanishing gradients in deep networks)
2. **Easier optimization**: Model can learn identity function trivially (do nothing), then incrementally add features
3. **Information preservation**: Early layer features always accessible to later layers

Think of residual stream as a "tape" that each layer reads from and writes to. Layers communicate by adding to the stream.

---

## Full Architecture

**Forward pass:**

1. Token embeddings + positional embeddings → residual stream initialization
2. For each of $L$ transformer blocks:
   - Apply LayerNorm, run multi-head attention, add to stream
   - Apply LayerNorm, run MLP, add to stream
3. Final LayerNorm
4. Unembed: project to vocabulary via $W_U \in \mathbb{R}^{d_{\text{model}} \times d_{\text{vocab}}}$
5. Get logits $\in \mathbb{R}^{n \times d_{\text{vocab}}}$

**Mathematically:**

$$
x^{(0)} = W_E[\text{tokens}] + W_P[\text{positions}]
$$

For $\ell = 1, \ldots, L$:

$$
x^{(\ell)} = x^{(\ell-1)} + \text{Attn}^{(\ell)}(\text{LN}(x^{(\ell-1)})) + \text{MLP}^{(\ell)}(\text{LN}(x^{(\ell-1)} + \text{Attn}^{(\ell)}(\cdots)))
$$

$$
\text{logits} = \text{LN}(x^{(L)}) W_U
$$

**GPT-2 configuration:**

- $L = 12$ layers
- $d_{\text{model}} = 768$
- $d_{\text{head}} = 64$
- $h = 12$ heads
- $d_{\text{mlp}} = 3072$
- $d_{\text{vocab}} = 50257$
- $n_{\text{ctx}} = 1024$

**Parameter count:**

- Embeddings: $50257 \times 768 + 1024 \times 768 \approx 39.4M$
- Each layer: $(4 \times 768 \times 768) + (768 \times 3072) + (3072 \times 768) + \text{biases} \approx 7.1M$
- Total: $39.4M + 12 \times 7.1M \approx 124M$ parameters

---

## Implementation

Clean PyTorch implementation following the math:

```python
import torch as t
import torch.nn as nn
from einops import einsum, rearrange

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(t.ones(d_model))
        self.b = nn.Parameter(t.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.w * (x - mean) / (var + self.eps).sqrt() + self.b


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_vocab, d_model) * 0.02)

    def forward(self, tokens):
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, n_ctx, d_model):
        super().__init__()
        self.W_P = nn.Parameter(t.randn(n_ctx, d_model) * 0.02)

    def forward(self, tokens):
        n = tokens.shape[-1]
        return self.W_P[:n]


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

        self.W_Q = nn.Parameter(t.randn(n_heads, d_model, d_head) * 0.02)
        self.W_K = nn.Parameter(t.randn(n_heads, d_model, d_head) * 0.02)
        self.W_V = nn.Parameter(t.randn(n_heads, d_model, d_head) * 0.02)
        self.W_O = nn.Parameter(t.randn(n_heads, d_head, d_model) * 0.02)

        self.b_Q = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_K = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_V = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_O = nn.Parameter(t.zeros(d_model))

        self.register_buffer("mask", t.tril(t.ones(1024, 1024)))

    def forward(self, x):
        Q = einsum(x, self.W_Q, "b n d, h d dh -> b n h dh") + self.b_Q
        K = einsum(x, self.W_K, "b n d, h d dh -> b n h dh") + self.b_K
        V = einsum(x, self.W_V, "b n d, h d dh -> b n h dh") + self.b_V

        scores = einsum(Q, K, "b qi h dh, b ki h dh -> b h qi ki") / (self.d_head ** 0.5)

        n = x.shape[1]
        mask = self.mask[:n, :n]
        scores = t.where(mask.bool(), scores, -1e9)

        pattern = scores.softmax(dim=-1)

        out = einsum(pattern, V, "b h qi ki, b ki h dh -> b qi h dh")
        out = einsum(out, self.W_O, "b n h dh, h dh d -> b n d") + self.b_O

        return out


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Parameter(t.randn(d_model, d_mlp) * 0.02)
        self.W_out = nn.Parameter(t.randn(d_mlp, d_model) * 0.02)
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.b_out = nn.Parameter(t.zeros(d_model))

    def forward(self, x):
        x = einsum(x, self.W_in, "b n d, d m -> b n m") + self.b_in
        x = gelu(x)
        x = einsum(x, self.W_out, "b n m, m d -> b n d") + self.b_out
        return x


def gelu(x):
    return 0.5 * x * (1 + t.tanh(((2 / t.pi) ** 0.5) * (x + 0.044715 * x ** 3)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, n_heads, d_head):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, d_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Unembed(nn.Module):
    def __init__(self, d_model, d_vocab):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab) * 0.02)

    def forward(self, x):
        return einsum(x, self.W_U, "b n d, d v -> b n v")


class Transformer(nn.Module):
    def __init__(self, d_model=768, n_layers=12, n_heads=12,
                 d_mlp=3072, d_vocab=50257, n_ctx=1024):
        super().__init__()
        d_head = d_model // n_heads

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_mlp, n_heads, d_head)
            for _ in range(n_layers)
        ])
        self.ln_final = LayerNorm(d_model)
        self.unembed = Unembed(d_model, d_vocab)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.unembed(x)
```

---

## Training

### Loss Function

**General cross-entropy** between distributions $p$ (true) and $q$ (predicted):

$$
H(p, q) = -\sum_{x} p(x) \log q(x)
$$

Also equivalent to KL divergence plus entropy:

$$
H(p, q) = D_{\text{KL}}(p \| q) + H(p)
$$

Since $H(p)$ is constant (doesn't depend on model), minimizing cross-entropy = minimizing KL divergence.

**For language modeling:** True distribution $p$ is one-hot at correct token $x_{i+1}$. So $p(x_{i+1}) = 1$, $p(k) = 0$ for $k \neq x_{i+1}$.

Cross-entropy simplifies:

$$
H(p, q) = -\sum_{k} p(k) \log q(k) = -1 \cdot \log q(x_{i+1}) = -\log q(x_{i+1})
$$

where $q(x_{i+1}) = \text{softmax}(\text{logits}_i)[x_{i+1}]$ is model's predicted probability for correct token.

**Average over sequence** $[x_1, \ldots, x_n]$:

$$
\mathcal{L} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log p_\theta(x_{i+1} \mid x_1, \ldots, x_i)
$$

Minimizing this = maximizing log-likelihood of data.

### Optimization

Standard setup:
- Optimizer: AdamW
- Learning rate: $3 \times 10^{-4}$ with warmup + cosine decay
- Batch size: 32-128 sequences
- Gradient clipping: norm 1.0
- Weight decay: 0.01

**Training loop:**

```python
def train(model, data_loader, epochs=10):
    opt = t.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for epoch in range(epochs):
        for tokens in data_loader:
            tokens = tokens.to(device)

            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
```

---

## Text Generation

### Greedy Sampling

Take most likely token at each step:

$$
x_{t+1} = \arg\max_i \; \text{logits}_t[i]
$$

**Problem:** Deterministic, repetitive. Gets stuck in loops.

### Temperature Sampling

Scale logits before softmax:

$$
p(x) = \text{softmax}(\text{logits} / T)
$$

where $T \in (0, \infty)$ is temperature.

**Effect:**
- $T \to 0$: sharpens distribution (approaches greedy)
- $T = 1$: unchanged distribution
- $T \to \infty$: uniform distribution (maximum randomness)

**Why it works:**

Scaling logits changes relative probabilities. If $\text{logits} = [5, 3, 1]$:
- $T = 0.5$: $[10, 6, 2] \to$ probabilities $[0.997, 0.003, 0.000]$ (sharp)
- $T = 1.0$: $[5, 3, 1] \to$ probabilities $[0.843, 0.155, 0.002]$ (original)
- $T = 2.0$: $[2.5, 1.5, 0.5] \to$ probabilities $[0.666, 0.242, 0.091]$ (smooth)

```python
def sample_with_temperature(logits, T=1.0):
    probs = (logits / T).softmax(dim=-1)
    return t.multinomial(probs, num_samples=1)
```

### Top-k Sampling

Only sample from $k$ most likely tokens. Set all other logits to $-\infty$.

$$
\text{logits}'[i] = \begin{cases}
\text{logits}[i] & \text{if } i \in \text{top-k}(\text{logits}) \\
-\infty & \text{otherwise}
\end{cases}
$$

Then apply softmax to $\text{logits}'$ and sample.

**Why it works:**

Prevents sampling from long tail of low-probability tokens that might be nonsensical. Keeps diversity while avoiding garbage.

```python
def sample_top_k(logits, k=50):
    top_logits, top_indices = logits.topk(k)
    probs = top_logits.softmax(dim=-1)
    sampled = t.multinomial(probs, num_samples=1)
    return top_indices.gather(-1, sampled)
```

### Top-p Sampling (Nucleus)

Sample from smallest set of tokens whose cumulative probability exceeds $p$.

**Algorithm:**
1. Sort tokens by probability (descending)
2. Compute cumulative sum of probabilities
3. Keep tokens until cumsum $\geq p$
4. Renormalize and sample from this subset

**Why better than top-k?**

Adaptive: uses fewer tokens when distribution is sharp (model is confident), more tokens when distribution is flat (model is uncertain).

```python
def sample_top_p(logits, p=0.9):
    probs = logits.softmax(dim=-1)
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    mask = cumsum <= p
    mask[..., 0] = True

    filtered_probs = t.where(mask, sorted_probs, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled = t.multinomial(filtered_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled)
```

### Beam Search

Maintain $k$ most likely sequences at each step.

**Algorithm:**
1. Start with $k$ beams (initially just start token)
2. For each beam, compute scores for all possible next tokens
3. Keep top $k$ beam-token pairs by cumulative log probability
4. Repeat until max length or all beams end

**Scoring:**

For beam with sequence $[x_1, \ldots, x_t]$ and next token $x_{t+1}$:

$$
\text{score}([x_1, \ldots, x_{t+1}]) = \sum_{i=1}^{t+1} \log p(x_i \mid x_{<i})
$$

Often use length normalization:

$$
\text{score} = \frac{1}{|x|^\alpha} \sum_{i=1}^{|x|} \log p(x_i \mid x_{<i})
$$

where $\alpha \in [0, 1]$ (typically 0.6-0.7).

**Why beam search?**

Greedy picks best token at each step (local optimum). Beam search explores multiple hypotheses simultaneously, often finds better global solutions.

**Tradeoff:** More computation (tracking $k$ beams), but higher quality outputs.

```python
def beam_search(model, prompt_tokens, k=4, max_len=50):
    beams = [(prompt_tokens, 0.0)]

    for _ in range(max_len):
        candidates = []

        for seq, score in beams:
            if seq[-1] == eos_token:
                candidates.append((seq, score))
                continue

            logits = model(seq.unsqueeze(0))[0, -1]
            log_probs = logits.log_softmax(dim=-1)

            top_log_probs, top_tokens = log_probs.topk(k)

            for log_prob, token in zip(top_log_probs, top_tokens):
                new_seq = t.cat([seq, token.unsqueeze(0)])
                new_score = score + log_prob.item()
                candidates.append((new_seq, new_score))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

        if all(seq[-1] == eos_token for seq, _ in beams):
            break

    return beams[0][0]
```

---

## Example Outputs

### Greedy Sampling

**Prompt:** "The transformer architecture"

**Output:**
```
The transformer architecture is a neural network architecture that is used to
train neural networks. The transformer architecture is a neural network
architecture that is used to train neural networks. The transformer
```

Repetitive. Gets stuck in loops.

---

### Temperature = 0.7

**Prompt:** "The transformer architecture"

**Output:**
```
The transformer architecture is based on self-attention mechanisms that allow
the model to weigh the importance of different words in a sequence. This
enables better context understanding and has revolutionized natural language
processing tasks.
```

More diverse, coherent.

---

### Top-p = 0.9, Temperature = 0.8

**Prompt:** "Once upon a time"

**Output:**
```
Once upon a time, there lived a young girl named Alice who discovered a
mysterious garden behind her house. The garden was filled with flowers that
glowed in the moonlight, and strange creatures that spoke in riddles. She
knew this place would change her life forever.
```

Creative, maintains coherence.

---

## Links & Resources

**Original Paper:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

**GPT Series:**
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)

**Implementations & Tutorials:**
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (Rush et al.)
- [minGPT](https://github.com/karpathy/minGPT) (Karpathy)
- [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy)

**Deep Dives:**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Alammar)
- [Transformer Circuits Thread](https://transformer-circuits.pub/) (Anthropic)

---

## KV Cache

**Problem:** During generation, we recompute attention for all previous tokens at every step. Wasteful.

For token $t$, attention computes:

$$
Q_t = x_t W_Q, \quad K_{1:t} = x_{1:t} W_K, \quad V_{1:t} = x_{1:t} W_V
$$

$$
\text{out}_t = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_{\text{head}}}}\right) V_{1:t}
$$

At step $t+1$, we compute $K_{1:t+1}$ and $V_{1:t+1}$ from scratch. But $K_{1:t}$ and $V_{1:t}$ haven't changed!

**Solution:** Cache previous keys and values.

At step $t$:
1. Load cached $K_{1:t-1}$, $V_{1:t-1}$ from memory
2. Compute only $K_t = x_t W_K$, $V_t = x_t W_V$
3. Concatenate: $K_{1:t} = [K_{1:t-1}; K_t]$, $V_{1:t} = [V_{1:t-1}; V_t]$
4. Store updated cache
5. Compute attention as normal

**Complexity:**
- Without cache: $O(t \cdot d_{\text{model}})$ per step → $O(T^2 \cdot d_{\text{model}})$ total for sequence length $T$
- With cache: $O(d_{\text{model}})$ per step → $O(T \cdot d_{\text{model}})$ total

**Memory tradeoff:** Store $2 \times L \times T \times d_{\text{model}}$ values (keys + values, all layers, all positions). For GPT-2: $2 \times 12 \times 1024 \times 768 \approx 19M$ floats per sequence.

Worth it for generation (which is sequential). Not needed for training (which processes full sequences in parallel).

```python
class AttentionWithCache(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

        self.W_Q = nn.Parameter(t.randn(n_heads, d_model, d_head) * 0.02)
        self.W_K = nn.Parameter(t.randn(n_heads, d_model, d_head) * 0.02)
        self.W_V = nn.Parameter(t.randn(n_heads, d_model, d_head) * 0.02)
        self.W_O = nn.Parameter(t.randn(n_heads, d_head, d_model) * 0.02)

        self.b_Q = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_K = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_V = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_O = nn.Parameter(t.zeros(d_model))

        self.register_buffer("mask", t.tril(t.ones(1024, 1024)))

    def forward(self, x, cache=None):
        Q = einsum(x, self.W_Q, "b n d, h d dh -> b n h dh") + self.b_Q
        K = einsum(x, self.W_K, "b n d, h d dh -> b n h dh") + self.b_K
        V = einsum(x, self.W_V, "b n d, h d dh -> b n h dh") + self.b_V

        if cache is not None:
            K_cache, V_cache = cache
            K = t.cat([K_cache, K], dim=1)
            V = t.cat([V_cache, V], dim=1)

        scores = einsum(Q, K, "b qi h dh, b ki h dh -> b h qi ki") / (self.d_head ** 0.5)

        n = K.shape[1]
        mask = self.mask[:n, :n]
        scores = t.where(mask.bool(), scores, -1e9)

        pattern = scores.softmax(dim=-1)
        out = einsum(pattern, V, "b h qi ki, b ki h dh -> b qi h dh")
        out = einsum(out, self.W_O, "b n h dh, h dh d -> b n d") + self.b_O

        return out, (K, V)
```

Generation loop with cache:

```python
def generate_with_cache(model, prompt_tokens, max_len=50):
    tokens = prompt_tokens
    caches = [None] * len(model.blocks)

    for _ in range(max_len):
        x = model.embed(tokens) + model.pos_embed(tokens)

        for i, block in enumerate(model.blocks):
            x_norm = block.ln1(x)
            attn_out, caches[i] = block.attn(x_norm[:, -1:], cache=caches[i])
            x = t.cat([x[:, :-1], x[:, -1:] + attn_out], dim=1)

            x_norm = block.ln2(x)
            mlp_out = block.mlp(x_norm[:, -1:])
            x = t.cat([x[:, :-1], x[:, -1:] + mlp_out], dim=1)

        logits = model.unembed(model.ln_final(x[:, -1:]))
        next_token = logits.argmax(dim=-1)
        tokens = t.cat([tokens, next_token], dim=1)

        if next_token == eos_token:
            break

    return tokens
```
