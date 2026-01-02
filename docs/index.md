---
title: Home
layout: default
nav_order: 1
permalink: /
---

# Juan’s ML Notes

This site collects my notes + implementations (math included).

## Backpropagation from Scratch

```python
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator
import numpy as np

Arr = np.ndarray
grad_tracking_enabled = True
```

We’ll implement backpropagation from scratch.  
For an intuitive intro, see [Colah’s post](https://colah.github.io/posts/2015-08-Backprop/).

For example, the backward function for `log` is:

```python
def log_back(grad_out, out, x):
    return grad_out / x
```

### Broadcasting in the backward pass

In general, we work with tensors. Suppose

- `out = x + y`
- `x.shape = (2,)`
- `y.shape = (4, 2)`

Under the hood, NumPy **[broadcasts](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)** `x` to a broadcasted version \(x_b\) with the shape of \(y\), then computes:

\[
\text{out} = x_b + y.
\]

How do we go from \(\frac{\partial L}{\partial x_b}\) to \(\frac{\partial L}{\partial x}\)?

Let \(x \in \mathbb{R}^d\), and let \(x_b\) be its broadcasted version with:

\[
x_b[i_1,\dots,i_k,j] = x[j].
\]

Let \(L = L(x_b)\). Then for each \(j\),

\[
\frac{\partial L}{\partial x[j]}
= \sum_{i_1,\dots,i_k}
\frac{\partial L}{\partial x_b[i_1,\dots,i_k,j]}
\cdot
\frac{\partial x_b[i_1,\dots,i_k,j]}{\partial x[j]}.
\]

Since

\[
\frac{\partial x_b[i_1,\dots,i_k,j]}{\partial x[j]} = 1,
\]

we obtain the reduction rule:

\[
\boxed{
\frac{\partial L}{\partial x[j]}
=
\sum_{i_1,\dots,i_k}
\frac{\partial L}{\partial x_b[i_1,\dots,i_k,j]}
}
\]

Equivalently:

\[
\boxed{
\nabla_x L
=
\sum_{\text{broadcast axes}}
\nabla_{x_b} L
}
\]

Here’s a practical implementation:

```python
def unbroadcast(broadcasted, original):
    b = broadcasted

    # If original had fewer dims, sum out the leading broadcast dims
    ndims = b.ndim - original.ndim
    if ndims > 0:
        b = b.sum(axis=tuple(range(ndims)))

    # For any axis where original had size 1 but b has size > 1, sum over that axis
    if original.ndim > 0:
        dims = tuple(
            i for i, (o, bi) in enumerate(zip(original.shape, b.shape))
            if o == 1 and bi != 1
        )
        if len(dims) > 0:
            b = b.sum(axis=dims, keepdims=True)

    assert b.shape == original.shape
    return b
```

A function can be differentiable w.r.t. more than one input tensor, so we often need a separate backward function per argument.

For example, for \(x * y\), the backward functions w.r.t. argument 0 and 1 are:

```python
def multiply_back0(grad_out, out, x, y):
    if not isinstance(y, Arr):
        y = np.array(y)
    return unbroadcast(grad_out * y, x)

def multiply_back1(grad_out, out, x, y):
    if not isinstance(x, Arr):
        x = np.array(x)
    return unbroadcast(grad_out * x, y)
```

### Manual backprop on a small graph

Let’s do backprop on this computational graph:

![Computational Graph]({{ "/assets/images/backpropexample.png" | relative_url }})

```python
def forward_and_back(a, b, c):
    d = np.multiply(a, b)
    e = np.log(c)
    f = np.multiply(d, e)
    g = np.log(f)

    go = np.ones_like(g)

    gf = log_back(go, g, f)
    gd = multiply_back0(gf, f, d, e)
    ge = multiply_back1(gf, f, d, e)

    ga = multiply_back0(gd, d, a, b)
    gb = multiply_back1(gd, d, a, b)
    gc = log_back(ge, e, c)

    return ga, gb, gc
```

Now, rather than manually figuring out which backward functions to call (and in what order), we’ll automate the process.

## Tracking the computation graph

We’ll store forward-pass metadata in a `Recipe` object, so we can compute gradients during backprop.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Recipe:
    func: object
    args: tuple
    kwargs: dict
    parents: dict
```

Next we need a way to look up the right backward function for a forward function + argument position.

```python
class BackwardFuncLookup:
    def __init__(self):
        self.back_funcs = {}

    def add_back_func(self, forward_fn, arg_position, backward_fn):
        self.back_funcs[(forward_fn, arg_position)] = backward_fn

    def get_back_func(self, forward_fn, arg_position):
        key = (forward_fn, arg_position)
        if key not in self.back_funcs:
            raise KeyError(f"no backward registered for {forward_fn.__name__} arg {arg_position}")
        return self.back_funcs[key]
```

Register the functions we already made:

```python
BACK_FUNCS = BackwardFuncLookup()

BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)
```

## A Tensor wrapper

We’ll wrap NumPy arrays in a `Tensor` class (similar to `torch.Tensor`).

Two key ideas:

1. **Leaf tensors** are “ends” of the backprop path: either they have no parents (no recipe / no upstream), or they don’t require grad. We store gradients only for leaf nodes that `requires_grad=True`.

2. `requires_grad` for outputs should be true iff all of these hold:
   - global grad tracking is enabled,
   - at least one input requires grad,
   - the function is differentiable.

```python
class Tensor:
    def __init__(self, array, requires_grad=False):
        self.array = array if isinstance(array, np.ndarray) else np.array(array)
        if self.array.dtype == np.float64:
            self.array = self.array.astype(np.float32)

        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None

    def __repr__(self):
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __hash__(self):
        return id(self)

    def __len__(self):
        if self.array.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.array.shape[0]

    def item(self):
        return self.array.item()

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        return not (self.requires_grad and self.recipe and self.recipe.parents)

    def __bool__(self):
        if int(np.prod(self.shape)) != 1:
            raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
        return bool(self.item())

    def __neg__(self):
        return negative(self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __eq__(self, other):
        return eq(self, other)

    def __getitem__(self, index):
        return getitem(self, index)

    def add_(self, other, alpha=1.0):
        add_(self, other, alpha=alpha)
        return self

    def sub_(self, other, alpha=1.0):
        sub_(self, other, alpha=alpha)
        return self

    def __iadd__(self, other):
        return self.add_(other)

    def __isub__(self, other):
        return self.sub_(other)

    @property
    def T(self):
        return permute(self, axes=(-1, -2))

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def permute(self, dims):
        return permute(self, axes=dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low, high):
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad=None):
        if isinstance(end_grad, np.ndarray):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)


def empty(*shape):
    return Tensor(np.empty(shape))


def zeros(*shape):
    return Tensor(np.zeros(shape))


def arange(start, end, step=1):
    return Tensor(np.arange(start, end, step=step))


def tensor(array, requires_grad=False):
    return Tensor(array, requires_grad=requires_grad)
```

## Wrapping NumPy functions

We want the “Tensor version” of NumPy functions like `np.log`, `np.multiply`, etc.

Here’s one explicit example (log):

```python
def log_forward(x):
    out_array = np.log(x.array)
    requires_grad = grad_tracking_enabled and x.requires_grad
    out = Tensor(out_array, requires_grad=requires_grad)

    if requires_grad:
        out.recipe = Recipe(func=np.log, args=(x.array,), kwargs={}, parents={0: x})

    return out
```

Multiply is similar, but one input might be a scalar:

```python
def multiply_forward(a, b):
    if not (isinstance(a, Tensor) or isinstance(b, Tensor)):
        raise AssertionError

    a_is_t = isinstance(a, Tensor)
    b_is_t = isinstance(b, Tensor)

    a_val = a.array if a_is_t else a
    b_val = b.array if b_is_t else b

    out_arr = a_val * b_val
    if not isinstance(out_arr, np.ndarray):
        out_arr = np.array(out_arr)

    req = False
    if grad_tracking_enabled:
        if (a_is_t and a.requires_grad) or (b_is_t and b.requires_grad):
            req = True

    out = Tensor(out_arr, req)

    if req:
        parents = {}
        if a_is_t:
            parents[0] = a
        if b_is_t:
            parents[1] = b
        out.recipe = Recipe(np.multiply, (a_val, b_val), {}, parents)

    return out
```

Rather than writing this boilerplate for every function, we can write a higher-order wrapper.

```python
def wrap_forward_fn(numpy_func, is_differentiable=True):
    def tensor_func(*args, **kwargs):
        raw = []
        need_grad = False

        for x in args:
            if isinstance(x, Tensor):
                raw.append(x.array)
                if x.requires_grad:
                    need_grad = True
            else:
                raw.append(x)

        out_arr = numpy_func(*raw, **kwargs)

        track = bool(grad_tracking_enabled and is_differentiable and need_grad)
        out = Tensor(out_arr, track)

        if track:
            parents = {}
            for i, x in enumerate(args):
                if isinstance(x, Tensor):
                    parents[i] = x
            out.recipe = Recipe(numpy_func, tuple(raw), dict(kwargs), parents)

        return out

    return tensor_func


def _sum(x, dim=None, keepdim=False):
    return np.sum(x, axis=dim, keepdims=keepdim)


log = wrap_forward_fn(np.log)
multiply = wrap_forward_fn(np.multiply)
eq = wrap_forward_fn(np.equal, is_differentiable=False)
sum = wrap_forward_fn(_sum)
```

## Reverse topological order of the graph

Backprop requires processing nodes from outputs back to inputs. One clean way to do this is to traverse the computation graph in **reverse topological order**.

A *topological ordering* of a directed acyclic graph (DAG) is an ordering where every node appears **after** all of its dependencies.

Below is a DFS-based topological sort which appends a node only after visiting all of its children:

```python
class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node):
    return node.children


def topological_sort(root, get_children):
    out = []
    done = set()
    active = set()

    def dfs(v):
        if v in done:
            return
        if v in active:
            raise ValueError("cycle detected")
        active.add(v)
        for u in get_children(v):
            dfs(u)
        active.remove(v)
        done.add(v)
        out.append(v)

    dfs(root)
    return out
```

For our tensors, we’ll treat a tensor’s “children” as its **parents** in the computation graph (the inputs to the operation that produced it). Then a topological ordering will naturally visit inputs before outputs.

```python
def sorted_computational_graph(tensor):
    def parents(t):
        if t.recipe is None or not t.recipe.parents:
            return []
        return list(t.recipe.parents.values())

    return topological_sort(tensor, parents)[::-1]
```

We return `[::-1]` because the DFS ordering places the end node last; for backprop we want to start from the end node and go backwards.

## Seeding backprop when the output isn’t scalar

Let the final node be a tensor \(g \in \mathbb{R}^{d_1 \times \cdots \times d_k}\) (not necessarily scalar). To define a scalar objective, pick a weight tensor \(v\) with the same shape and define:

\[
L \;=\; \langle g, v\rangle
\;=\;
\sum_{i_1,\dots,i_k} g_{i_1,\dots,i_k} v_{i_1,\dots,i_k}
\;=\; (g * v).sum().
\]

Then the seed gradient is:

\[
\frac{\partial L}{\partial g_{i_1,\dots,i_k}} = v_{i_1,\dots,i_k}
\quad\Longrightarrow\quad
\nabla_g L = v.
\]

So, in backprop, the **first** `grad_out` passed into the backward function at the end node \(g\) is exactly \(v\).

Default behavior: take \(v=\mathbf{1}\) (all ones), which corresponds to \(L = g.sum()\).

If you pass `end_grad`, you are explicitly choosing \(v := \texttt{end\_grad}\).

## Backprop implementation

```python
def backprop(end_node, end_grad=None):
    end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array
    if not isinstance(end_grad_arr, np.ndarray):
        end_grad_arr = np.array(end_grad_arr)

    grads = {end_node: end_grad_arr}

    for node in sorted_computational_graph(end_node):
        outgrad = grads.pop(node)

        if node.is_leaf:
            if node.requires_grad:
                if node.grad is None:
                    node.grad = Tensor(outgrad)
                else:
                    node.grad.array += outgrad
            continue

        recipe = node.recipe
        assert recipe is not None

        for argnum, parent in recipe.parents.items():
            back_fn = BACK_FUNCS.get_back_func(recipe.func, argnum)
            in_grad = back_fn(outgrad, node.array, *recipe.args, **recipe.kwargs)
            grads[parent] = grads[parent] + in_grad if parent in grads else in_grad

    return None
```

## Adding more backward functions

Negative:

```python
def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return -grad_out

BACK_FUNCS.add_back_func(np.negative, 0, negative_back)
```

Exponential:

```python
def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return grad_out * out

BACK_FUNCS.add_back_func(np.exp, 0, exp_back)
```

Reshape: backward just reshapes the gradient back to the original shape.

```python
def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return np.reshape(grad_out, x.shape)

BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)
```

Permute / transpose: backward permutes by the inverse permutation.

```python
def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, np.argsort(axes))

BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
```

## Backward for `sum`

Sometimes the output is smaller than the input, e.g. `sum`.

Let \(x\in\mathbb{R}^{n_1\times\cdots\times n_k}\). Fix axes \(D\subseteq\{1,\dots,k\}\) to sum over, and define \(y=\operatorname{sum}(x;D)\). Each \(y_j\) is a sum of many \(x_i\)’s that “collapse” to the same surviving index \(j\).

Take a scalar loss \(L=L(y)\). For any entry \(x_i\) contributing to \(y_j\),

\[
\frac{\partial y_j}{\partial x_i} = 1.
\]

So by the chain rule,

\[
\frac{\partial L}{\partial x_i}
=
\frac{\partial L}{\partial y_j}.
\]

In words: **every element that got summed into the same output entry receives the same upstream gradient value.** This is exactly “broadcasting the output gradient back to the input shape”.

Implementation-wise, `sum_back` is just:

1. If `keepdim=False`, re-insert the summed axes as size-1 dimensions so broadcasting works.
2. Broadcast to `x.shape`.

(If `dim=None`, then `y` is scalar and this still broadcasts correctly.)
