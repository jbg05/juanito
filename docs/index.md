---
title: Home
layout: default
nav_order: 1
---

# Juan’s ML Notes

This site collects my notes + implementations (math included).

# Backpropagation from Scratch

```python
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator
import numpy as np

Arr = np.ndarray
grad_tracking_enabled = True
```

We will implement backpropagation from scratch.  
Read [this](https://colah.github.io/posts/2015-08-Backprop/) for a basic understanding of backprop.

So, for example our backwards function of log would be:

```python
def log_back(grad_out, out, x):
    return grad_out / x
```

In general, we are working with tensors, where we might have something like out=x+y where x.shape=(2,) and y.shape=(4,2), then whats happening under the hood is we broadcast (see https://docs.pytorch.org/docs/stable/notes/broadcasting.html) to some x_b which has the shape of y and then define out=x_b+y

Now, how do we go from dL/d(x_b) to finding dL/dx?

Let \(x \in \mathbb{R}^d\) and let \(x_b\) be its broadcasted version with
\[
x_b[i_1,\dots,i_k,j] = x[j].
\]

Let \(L = L(x_b)\). Then for each \(j\),
\[
\frac{\partial L}{\partial x[j]}
=
\sum_{i_1,\dots,i_k}
\frac{\partial L}{\partial x_b[i_1,\dots,i_k,j]}
\frac{\partial x_b[i_1,\dots,i_k,j]}{\partial x[j]}.
\]

Since
\[
\frac{\partial x_b[i_1,\dots,i_k,j]}{\partial x[j]} = 1,
\]
we obtain
\[
\boxed{
\frac{\partial L}{\partial x[j]}
=
\sum_{i_1,\dots,i_k}
\frac{\partial L}{\partial x_b[i_1,\dots,i_k,j]}
}.
\]

Equivalently,
\[
\boxed{
\nabla_x L = \sum_{\text{broadcast axes}} \nabla_{x_b} L.
}
\]

