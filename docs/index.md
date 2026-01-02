---
layout: home
title: "Juan’s Notes"
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

If you pass `end_grad`, you are explicitly choosing \(v := \mathrm{end\_grad}\) (i.e., the array you passed in).

---

### Backprop implementation (with one quick explanation)

The key idea in the code below is:

- `grads[t]` stores the accumulated gradient \(\partial L/\partial t\) for each `Tensor` \(t\).
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

---

## Backward for `sum`

Now, there are cases where the output might be smaller than the input, like when using the sum function.

Let \(x \in \mathbb{R}^{n_1 \times \cdots \times n_k}\). Fix axes \(D \subseteq \{1,\dots,k\}\) to sum over, and define:

$$
y=\mathrm{sum}(x;D),\qquad 
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



```python
def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False) -> Arr:
    # If keepdim=False, NumPy removed the summed axis/axes.
    # Put them back as size-1 dims so broadcasting works.
    if (not keepdim) and (dim is not None):
        grad_out = np.expand_dims(grad_out, dim)

    # Every entry of x that contributed to the same summed output
    # receives the same upstream gradient value, i.e. broadcast back.
    return np.broadcast_to(grad_out, x.shape)

BACK_FUNCS.add_back_func(_sum, 0, sum_back)
```

Now: elementwise adding, subtracting, and dividing.

Notice that in general, for \( \text{out} = f(x,y) \) and scalar loss \( L = L(\text{out}) \), with
\( g := \frac{\partial L}{\partial \text{out}} \), the chain rule gives:

$$
\frac{\partial L}{\partial x}
=
\mathrm{unbroadcast}\!\left(g \odot \frac{\partial f}{\partial x}(x,y),\, x\right),
\qquad
\frac{\partial L}{\partial y}
=
\mathrm{unbroadcast}\!\left(g \odot \frac{\partial f}{\partial y}(x,y),\, y\right).
$$

For example, in elementwise division \( \text{out} = \frac{x}{y} \):

$$
\frac{\partial L}{\partial x}
=
\mathrm{unbroadcast}\!\left(g \odot \frac{1}{y},\, x\right),
\qquad
\frac{\partial L}{\partial y}
=
\mathrm{unbroadcast}\!\left(g \odot \left(-\frac{x}{y^2}\right),\, y\right),
\quad
g:=\frac{\partial L}{\partial \text{out}}.
$$

```python
BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
BACK_FUNCS.add_back_func(np.add, 1, lambda grad_out, out, x, y: unbroadcast(grad_out, y))

BACK_FUNCS.add_back_func(np.subtract, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
BACK_FUNCS.add_back_func(np.subtract, 1, lambda grad_out, out, x, y: unbroadcast(-grad_out, y))

BACK_FUNCS.add_back_func(
    np.true_divide,
    0,
    lambda grad_out, out, x, y: unbroadcast(grad_out / y, x),
)
BACK_FUNCS.add_back_func(
    np.true_divide,
    1,
    lambda grad_out, out, x, y: unbroadcast(-grad_out * x / (y * y), y),
)
```

Now, the `maximum` function is pretty interesting.

For \( \max(x,y) \), the derivative w.r.t. \(x\) is \(1\) when \(x>y\) and \(0\) when \(x<y\). The only tricky case is \(x=y\).

At a tie, `max` is **not differentiable** in the strict sense (there isn’t a unique slope). But it *is* subdifferentiable: any “split” of the upstream gradient between the two arguments that sums to \(1\) is a valid choice.

Why should the splits sum to \(1\)? Intuitively, when \(x=y\), the function \(\max(x,y)\) behaves like the identity along the line \(x=y\): if you increase both inputs by the same small amount \(t\), the output increases by \(t\). So the total sensitivity to moving along \((1,1)\) should be \(1\). That corresponds to choosing partials \(\alpha\) and \(1-\alpha\) with \(\alpha\in[0,1]\). A common convention is \(\alpha=\tfrac12\), i.e. split evenly.

With `maximum`, `relu` follows.

```python
def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    mask = (x > y) + 0.5 * (x == y)
    return unbroadcast(grad_out * mask, x)

def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    mask = (x < y) + 0.5 * (x == y)
    return unbroadcast(grad_out * mask, y)

BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)


def relu(x: Tensor) -> Tensor:
    return maximum(x, 0.0)
```

Finally, we will take a look at 2D matrix multiplication and its backward methods (our in-house version of `torch.matmul`).

Let \(X\in\mathbb{R}^{n\times m}\), \(Y\in\mathbb{R}^{m\times k}\), and
\(M = XY \in \mathbb{R}^{n\times k}\), with \(L = L(M)\).
Define the upstream gradient

$$
G := \frac{\partial L}{\partial M}\in\mathbb{R}^{n\times k}
\quad\text{(this is `grad_out`).}
$$

Elementwise, for \(p\in[n]\) and \(q\in[k]\),

$$
M_{pq}=\sum_{r=1}^m X_{pr}Y_{rq}.
$$

---

## Gradient w.r.t. \(X\)

Fix \((i,j)\). By the chain rule:

$$
\frac{\partial L}{\partial X_{ij}}
=\sum_{p=1}^n\sum_{q=1}^k
\frac{\partial L}{\partial M_{pq}}
\frac{\partial M_{pq}}{\partial X_{ij}}
=\sum_{p,q} G_{pq}\frac{\partial}{\partial X_{ij}}
\Big(\sum_{r}X_{pr}Y_{rq}\Big).
$$

Use \(\dfrac{\partial X_{pr}}{\partial X_{ij}}=\mathbf{1}\{p=i,r=j\}\):

$$
\frac{\partial M_{pq}}{\partial X_{ij}} = Y_{jq}\,\mathbf{1}\{p=i\}.
$$

So

$$
\frac{\partial L}{\partial X_{ij}}=\sum_{q=1}^k G_{iq}Y_{jq} = (G Y^\top)_{ij}.
$$

Hence

$$
\boxed{\frac{\partial L}{\partial X}=G Y^\top}
\qquad\Longleftrightarrow\qquad
\boxed{\texttt{x.grad} = \texttt{grad\_out @ y.T}}.
$$

---

## Gradient w.r.t. \(Y\)

Fix \((j,q)\). By the chain rule:

$$
\frac{\partial L}{\partial Y_{jq}}
=\sum_{p=1}^n\sum_{t=1}^k
\frac{\partial L}{\partial M_{pt}}
\frac{\partial M_{pt}}{\partial Y_{jq}}
=\sum_{p,t} G_{pt}\frac{\partial}{\partial Y_{jq}}
\Big(\sum_{r}X_{pr}Y_{rt}\Big).
$$

Use \(\dfrac{\partial Y_{rt}}{\partial Y_{jq}}=\mathbf{1}\{r=j,t=q\}\):

$$
\frac{\partial M_{pt}}{\partial Y_{jq}}=X_{pj}\,\mathbf{1}\{t=q\}.
$$

So

$$
\frac{\partial L}{\partial Y_{jq}}=\sum_{p=1}^n G_{pq}X_{pj} = (X^\top G)_{jq}.
$$

Hence

$$
\boxed{\frac{\partial L}{\partial Y}=X^\top G}
\qquad\Longleftrightarrow\qquad
\boxed{\texttt{y.grad} = \texttt{x.T @ grad\_out}}.
$$

```python
def _matmul2d(x: Arr, y: Arr) -> Arr:
    return x @ y

def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return grad_out @ y.T

def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return x.T @ grad_out

matmul = wrap_forward_fn(_matmul2d)
BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)
```

For functions like `argmax` or `eq` there isn't a sensible way to define gradients since they aren't differentiable. But we still need to unbox the arguments and box the result, so we still use `wrap_forward_fn` and set `is_differentiable=False`.

```python
eq = wrap_forward_fn(np.equal, is_differentiable=False)

def _argmax(x: Arr, dim=None, keepdim=False):
    result = np.argmax(x, axis=dim)
    if keepdim:
        # If dim is None, argmax returns a scalar; expanding dims is a no-op.
        if dim is None:
            return np.expand_dims(result, axis=())
        return np.expand_dims(result, axis=dim)
    return result

argmax = wrap_forward_fn(_argmax, is_differentiable=False)
```

```python
log = wrap_forward_fn(np.log)
multiply = wrap_forward_fn(np.multiply)
add = wrap_forward_fn(np.add)
subtract = wrap_forward_fn(np.subtract)
true_divide = wrap_forward_fn(np.true_divide)
sum = wrap_forward_fn(_sum)
negative = wrap_forward_fn(np.negative)
exp = wrap_forward_fn(np.exp)
reshape = wrap_forward_fn(np.reshape)
permute = wrap_forward_fn(np.transpose)
maximum = wrap_forward_fn(np.maximum)
```


Now, we've seen enough backward passes to abstract further and write our own versions of `nn.Parameter` and `nn.Module`.

`Parameter` is just a `Tensor` with `requires_grad=True` by default.

```python
class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        super().__init__(tensor.array, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"
```

Below is our `nn.Module`. See the following if you're curious:

- [PyTorch `torch.nn.Module` documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Python Data Model (special method names)](https://docs.python.org/3/reference/datamodel.html)

```python
class Module:
    _modules: dict[str, "Module"]
    _parameters: dict[str, Parameter]

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self) -> Iterator["Module"]:
        yield from self._modules.values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from self._parameters.values()
        if recurse:
            for mod in self.modules():
                yield from mod.parameters(recurse=True)

    def __setattr__(self, key: str, val: Any) -> None:
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val
        super().__setattr__(key, val)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        _indent = lambda s_, nSpaces: re.sub("\n", "\n" + (" " * nSpaces), s_)
        lines = [f"({k}): {_indent(repr(m), 2)}" for k, m in self._modules.items()]
        return "".join(
            [
                self.__class__.__name__ + "(",
                "\n  " + "\n  ".join(lines) + "\n" if lines else "",
                ")",
            ]
        )
```

### Why do we do the `sf = 1/sqrt(d)` scaling?

This is a super standard “keep activations the same scale” argument.

Let $d=\text{in\_features}$ and $y=Wx$ with $W_{ji}$ i.i.d., $\mathbb{E}[W_{ji}]=0$, and $x_i$ roughly i.i.d. with $\mathbb{E}[x_i]=0$ and $\mathrm{Var}(x_i)\approx 1$. Then

$$
y_j=\sum_{i=1}^d W_{ji}x_i,
\qquad
\mathrm{Var}(y_j)\approx \sum_{i=1}^d \mathrm{Var}(W_{ji}x_i)
= d\,\mathrm{Var}(W_{ji}).
$$

To keep activations the same scale layer-to-layer (avoid exploding / vanishing), we want $\mathrm{Var}(y_j)\approx 1$, so set

$$
d\,\mathrm{Var}(W_{ji})\approx 1
\quad\Rightarrow\quad
\mathrm{Var}(W_{ji})\approx \frac{1}{d}
\quad\Rightarrow\quad
\mathrm{std}(W_{ji})\approx \frac{1}{\sqrt{d}}.
$$

Sampling $W_{ji}\sim \mathrm{Unif}[-sf,sf]$ with $sf \propto 1/\sqrt{d}$ implements this “variance-preserving” idea up to a constant factor (this is the same vibe as Xavier / He init).

---

We're going to try classifying MNIST, so we will define an MLP suitable for it.

```python
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 64)
        self.linear2 = Linear(64, 64)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.output = Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape((x.shape[0], 28 * 28))
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.output(x)
        return x
```

### Backward for indexing (`getitem`)

If we have the gradient of $L$ w.r.t. `x[index]`, what is the gradient of $L$ w.r.t. `x`?

It’s an array of zeros, except we “scatter-add” the upstream gradient values back into the positions selected by `index`. (If the same position is selected multiple times, the gradients should add.)

`index` can be an integer, a tuple of integers, or integer arrays. See NumPy’s docs on integer indexing [here](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing).

```python
def coerce_index(index):
    if isinstance(index, tuple):
        return tuple(i.array if hasattr(i, "array") else i for i in index)
    return index

def _getitem(x, index):
    return x[coerce_index(index)]

def getitem_back(grad_out, out, x, index):
    gx = np.zeros_like(x)
    np.add.at(gx, coerce_index(index), grad_out)
    return gx

getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)
```

Now that we can do integer-array indexing like `logprobs[arange(B), y]`, we can write cross-entropy in ~3 lines.

Let `logits = z ∈ R^{B×C}` and labels `y ∈ {0,…,C-1}^B`. Define the per-example loss

$$
\ell_i
= -\log\frac{e^{z_{i,y_i}}}{\sum_{c=0}^{C-1} e^{z_{i,c}}}
= -z_{i,y_i} + \log\Big(\sum_{c=0}^{C-1} e^{z_{i,c}}\Big).
$$

Equivalently, with log-softmax:

$$
\log p_{i,c} = z_{i,c} - \log\Big(\sum_{k} e^{z_{i,k}}\Big),
\qquad
\ell_i = -\log p_{i,y_i}.
$$

Vectorized form (left-to-right):

$$
\mathrm{logZ} = \log\Big(\sum_{c} e^{z_{\cdot,c}}\Big)\in\mathbb{R}^{B\times 1},
\quad
\mathrm{logprobs}=z-\mathrm{logZ}\in\mathbb{R}^{B\times C},
\quad
\ell = -\,\mathrm{logprobs}[\mathrm{arange}(B),\,y]\in\mathbb{R}^{B}.
$$

Quick gradient sanity (useful intuition): if $p=\mathrm{softmax}(z)$ and $e_{y_i}$ is one-hot, then

$$
\frac{\partial \ell_i}{\partial z_{i,\cdot}} = p_{i,\cdot}-e_{y_i}.
$$

So the correct class gets pushed up, others get pushed down, scaled by current probabilities.

*(If you care about stability later: replace $z$ by $z-\max_c z_{i,c}$ inside the $\log\sum e^{\cdot}$. Same math, fewer overflows.)*

```python
def cross_entropy(logits, true_labels):
    bsz = logits.shape[0]

    denom = logits.exp().sum(dim=1, keepdim=True).log()
    log_probs = logits - denom

    correct = log_probs[arange(0, bsz), true_labels]
    return -correct
```

---

### Turning gradients off (like `torch.inference_mode`)

The final thing our backprop system needs is the ability to turn off graph-building completely.

```python
class NoGrad:
    was_enabled: bool

    def __enter__(self):
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled
```

### SGD (Stochastic Gradient Descent)

SGD is the simplest optimizer: for each parameter $p$, do

$$
p \leftarrow p - \eta \,\nabla_p L,
$$

where $\eta$ is the learning rate.

- `zero_grad()` clears saved gradients from the previous step
- `step()` applies the update using the gradients we just computed with `backward()`

```python
class SGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            # p -= lr * grad
            p.sub_(Tensor(p.grad.array), alpha=self.lr)
```

At this point the hard part is done: we can do `forward → cross_entropy → backward → SGD.step`, so training is basically just plumbing.

Pipeline (left-to-right):

`MNIST DataLoader (torch tensors)` → `numpy()` → `Tensor(...)` → `model(x)=logits` → `loss = CE(logits,y)` → `loss.backward()` → `optimizer.step()`.

We’ll train for a few epochs, print a running loss, then evaluate with `NoGrad()` (same forward pass, but no graph / grads), and report test loss + accuracy. After this runs, you’ve got an end-to-end “from scratch” system: data → compute graph → backprop → parameter updates.

```python
import numpy as np

def get_mnist_loaders(batch_size=128, num_workers=0):
    # Returns (train_loader, test_loader) yielding (data, target) as torch tensors.
    # We convert to numpy inside the training loop.
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError(
            "You need torch + torchvision for this loader.\n"
            "Install with: pip install torch torchvision\n"
            f"Original import error: {e}"
        )

    tfm = transforms.Compose([transforms.ToTensor()])  # -> float in [0,1], shape (1,28,28)

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, epoch=0, log_every=200):
    total_loss = 0.0
    n_seen = 0

    for i, (data, target) in enumerate(train_loader):
        # data: torch.FloatTensor, shape (B,1,28,28)
        # target: torch.LongTensor, shape (B,)
        x = Tensor(data.numpy().astype(np.float32))
        y = Tensor(target.numpy().astype(np.int64))

        optimizer.zero_grad()
        logits = model(x)                                   # (B,10)
        loss = cross_entropy(logits, y).sum() / len(logits) # scalar Tensor
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(logits)
        n_seen += len(logits)

        if (i % log_every) == 0:
            print(f"epoch {epoch} | step {i:04d} | loss {loss.item():.4f}")

    print(f"epoch {epoch} train avg loss: {total_loss / max(n_seen,1):.4f}")


def eval_epoch(model, test_loader):
    correct = 0
    total = 0
    total_loss = 0.0

    with NoGrad():
        for data, target in test_loader:
            x = Tensor(data.numpy().astype(np.float32))
            y = Tensor(target.numpy().astype(np.int64))

            logits = model(x)
            loss_vec = cross_entropy(logits, y)  # shape (B,)
            total_loss += loss_vec.sum().item()

            pred = logits.argmax(dim=1, keepdim=False)  # shape (B,)
            correct += (pred == y).sum().item()
            total += len(y)

    avg_loss = total_loss / max(total,1)
    acc = correct / max(total,1)
    print(f"test avg loss: {avg_loss:.4f} | acc: {acc:.2%}")
    return avg_loss, acc


train_loader, test_loader = get_mnist_loaders(batch_size=128)

model = MLP()
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(5):
    train_epoch(model, train_loader, optimizer, epoch=epoch, log_every=200)
    eval_epoch(model, test_loader)
```

