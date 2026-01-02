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

We will implement backpropagation from scratch.  
For a friendly intro, read [this](https://colah.github.io/posts/2015-08-Backprop/).

So, for example, the backward function of `log` would be:

```python
def log_back(grad_out, out, x):
    return grad_out / x
```

## Broadcasting: how gradients “unbroadcast”

In general, we are working with tensors, where we might have something like `out = x + y` with:

- `x.shape = (2,)`
- `y.shape = (4, 2)`

Then what’s happening under the hood is NumPy **[broadcasts](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)** `x` to some \(x_b\) which has the shape of \(y\), and then defines:

$$
\text{out} = x_b + y.
$$

Now, how do we go from \(\frac{\partial L}{\partial x_b}\) to finding \(\frac{\partial L}{\partial x}\)?

Let \(x \in \mathbb{R}^d\) and let \(x_b\) be its broadcasted version with:

$$
x_b[i_1, \dots, i_k, j] = x[j].
$$

Let \(L = L(x_b)\). Then for each \(j\):

$$
\frac{\partial L}{\partial x[j]}
= \sum_{i_1, \dots, i_k}
\frac{\partial L}{\partial x_b[i_1, \dots, i_k, j]}
\cdot
\frac{\partial x_b[i_1, \dots, i_k, j]}{\partial x[j]}.
$$

Since the derivative of the broadcasted element with respect to the original is:

$$
\frac{\partial x_b[i_1, \dots, i_k, j]}{\partial x[j]} = 1,
$$

we obtain the final reduction formula:

$$
\boxed{
\frac{\partial L}{\partial x[j]}
=
\sum_{i_1, \dots, i_k}
\frac{\partial L}{\partial x_b[i_1, \dots, i_k, j]}
}.
$$

Equivalently, in gradient notation:

$$
\boxed{
\nabla_x L
=
\sum_{\text{broadcast axes}} \nabla_{x_b} L
}.
$$

```python
def unbroadcast(broadcasted, original):
    b = broadcasted

    # If original had fewer dims, sum out the leading broadcast dims.
    ndims = b.ndim - original.ndim
    if ndims > 0:
        b = b.sum(axis=tuple(range(ndims)))

    # For axes where original has size 1 but b has size > 1, sum over those axes.
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

Indeed, functions can be differentiable w.r.t. more than one input tensor, in which case we need multiple backward functions, one for each input argument.

For example, we can write the backward functions for \(x*y\) w.r.t. argument 0 and 1 (i.e., \(x\) and \(y\) respectively):

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

Now, let's try out our backward functions and do backpropagation on this computational graph:

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
Now, rather than figuring out which backward functions to call (and in what order, and with what inputs), we'll write code to automate that for us.

The class `Recipe` is necessary to track the forward functions in our computational graph so that we can calculate gradients during backprop.

```python
@dataclass(frozen=True, slots=True)
class Recipe:
    func: object
    args: tuple
    kwargs: dict
    parents: dict
```

While the `Recipe` class tracks the forward functions in our computational graph, we still need to find the backward functions corresponding to a given forward function automatically.

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

Let's register the functions we've already made for our example:

```python
BACK_FUNCS = BackwardFuncLookup()

BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)
```

Now we're going to wrap each array with a wrapper object needed for backpropagation. We'll call it `Tensor`, since it'll behave a lot like a `torch.Tensor`.

Most of this is very standard, except two important things:

1. A **leaf tensor** is one that represents the end of a backprop path: either it has no nodes further back which require gradients, or it doesn't require gradients itself. (This is important because our backprop algorithm will always stop at a leaf node.) We store the gradients of leaf nodes if `requires_grad` is true, unlike intermediate layers where storing the gradient is a waste of memory.

2. When creating tensors we can set `requires_grad` explicitly, but otherwise it's true iff:
   - (a) global grad tracking is enabled (we enabled this at the beginning of the article),
   - (b) at least one of the input tensors requires grad (otherwise there are no tensors further upstream),
   - (c) the function is differentiable (otherwise how could we even compute its gradients?).

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

Now, we can implement the functionality of NumPy functions but work with our newfound tensors instead of arrays. For example, for the `log` function:

```python
def log_forward(x):
    out_array = np.log(x.array)
    requires_grad = grad_tracking_enabled and x.requires_grad
    out = Tensor(out_array, requires_grad=requires_grad)

    if requires_grad:
        out.recipe = Recipe(func=np.log, args=(x.array,), kwargs={}, parents={0: x})

    return out
```

The multiply function is similar, but we need to be careful that one of the inputs may be an `int`:

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


With these two examples in mind, it's pretty straightforward to implement the higher-order function `wrap_forward_fn` that takes a NumPy function (roughly `Arr -> Arr`) and returns its `Tensor` equivalent as a `Callable`.

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

---

As part of backprop, we need to sort the nodes of our graph in **reverse topological order**.

A **topological sort** is a graph traversal where each node \(v\) is visited only after all its dependencies are visited (for every directed edge from \(u\) to \(v\), \(u\) comes before \(v\) in the ordering).

We need the reverse of this, since we propagate backward and calculate gradients of nodes only after we have the gradients of their downstream dependencies.

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

We can prove by induction that a cycle-free finite directed graph contains a topological ordering:

- If \(N = 1\), it's trivially true.
- If \(N > 1\), pick any node and follow outgoing edges until you reach a node with out-degree \(0\). This node must exist; otherwise you would either follow edges forever (contradicting finiteness) or revisit a node (contradicting acyclicity). Put this node first in your topological ordering, remove it from the graph, and apply topological sort to the remaining subgraph. By induction, that ordering exists. Prepending the removed node gives a topological ordering for all \(N\) vertices.

With its existence in mind, we can now safely take a tensor and return a list of tensors that make up its computational graph, in reverse topological order.

```python
def sorted_computational_graph(tensor):
    def parents(t):
        if t.recipe is None or not t.recipe.parents:
            return []
        return list(t.recipe.parents.values())

    return topological_sort(tensor, parents)[::-1]
```

---

Now we're fully ready to write our backprop function, but I'll clarify one thing first.

Let the final node be a tensor \(g \in \mathbb{R}^{d_1 \times \cdots \times d_k}\) (not necessarily scalar). To define “the” gradient, pick a weight tensor \(v\) with the same shape and define a scalar objective:

$$
L \;=\; \langle g, v\rangle
\;=\; \sum_{i_1,\dots,i_k} g_{i_1,\dots,i_k}\, v_{i_1,\dots,i_k}
\;=\; (g * v).sum().
$$

Then the seed gradient is:

$$
\frac{\partial L}{\partial g_{i_1,\dots,i_k}} \;=\; v_{i_1,\dots,i_k}
\quad\Longrightarrow\quad
\nabla_g L \;=\; v.
$$

So, in backprop, the **first** `grad_out` passed into the backward function at the end node \(g\) is exactly \(v\).

Special case (default behavior): take \(v=\mathbf{1}\) (all ones, same shape as \(g\)), giving:

$$
L = \sum_{i_1,\dots,i_k} g_{i_1,\dots,i_k} = g.sum(),
\qquad
\nabla_g L = \mathbf{1}.
$$

If you pass `end_grad`, you are explicitly choosing \(v := \texttt{end\_grad}\).

---

### Backprop implementation (with one quick explanation)

The key idea in the code below is:

- `grads[t]` stores the accumulated gradient \(\partial L/\partial t\) for each `Tensor` `t`.
- We process nodes from the end node backward.
- At each node, we push gradients to its parents using the registered backward functions.
- When we hit a leaf, we store (and accumulate) into `node.grad`.

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
```

Now that backprop is complete, we just need to add a couple backward functions.

```python
def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return -grad_out

BACK_FUNCS.add_back_func(np.negative, 0, negative_back)
```

```python
def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return grad_out * out

BACK_FUNCS.add_back_func(np.exp, 0, exp_back)
```

Reshape is a little different. The operation that takes us from \(\partial L/\partial x_r\) to \(\partial L/\partial x\) is exactly the inverse of the forward reshape operation that gave us \(x_r\) from \(x\), so we want to take `grad_out` and get it back to the shape of `x`.

```python
def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return np.reshape(grad_out, x.shape)

BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)
```

```python
def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, np.argsort(axes))

BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
```

Now, there are cases where the output might be smaller than the input, like when using the sum function.

Let \(x\in\mathbb{R}^{n_1\times\cdots\times n_k}\). Fix axes \(D\subseteq\{1,\dots,k\}\) to sum over, and define:

$$
y=\operatorname{sum}(x;D),\qquad 
y_{j}=\sum_{i\in I(j)} x_i,
$$

where \(j\) indexes the surviving coordinates (axes \(\notin D\)), and \(I(j)\) is the set of full indices \(i=(i_1,\dots,i_k)\) that collapse to the same \(j\) after summing over \(D\).

Take any scalar loss \(L=L(y)\). For any entry \(x_i\) with \(i\mapsto j\),

$$
\frac{\partial y_{j}}{\partial x_i}=1,\qquad 
\frac{\partial y_{j'}}{\partial x_i}=0\ \ (j'\neq j).
$$

Therefore, by the chain rule,

$$
\frac{\partial L}{\partial x_i}
=\sum_{j'}\frac{\partial L}{\partial y_{j'}}\frac{\partial y_{j'}}{\partial x_i}
=\frac{\partial L}{\partial y_j}.
$$

So every element \(x_i\) that was included in the same sum producing \(y_j\) receives the *same* upstream gradient value \(\big(\partial L/\partial y_j\big)\). In other words: the gradient \(\partial L/\partial y\) must be *replicated* across the axes that were summed out, to match the shape of \(x\). This replication is exactly a broadcast.

Recall that when we looked at broadcasting, the backward operation was summing over the broadcasted dimensions: broadcasting copies values, creating multiple identical computational paths, so gradients add across those copies. Summation is the “reverse”: the forward pass adds many \(x\)-entries into one \(y\)-entry, and the backward pass sends the same \(\partial L/\partial y_j\) back to *each* contributing \(x_i\). Hence, backward(\(\sum\)) = broadcast.

Implementation-wise, you can view `sum_back` as two purely shape-level steps:

- **(A) Reinsert summed axes when `keepdim=False`:** forward removed axes \(D\); backward inserts them back as size-1 dimensions so shapes are aligned.
- **(B) Broadcast to \(x\)’s full shape:** replicate \(\partial L/\partial y\) along exactly those axes \(D\), yielding an array with shape `x.shape`.

If `dim=None`, then \(y\) is a scalar and \(\partial L/\partial y\) is also scalar; broadcasting a scalar to `x`’s shape is still the same rule.

