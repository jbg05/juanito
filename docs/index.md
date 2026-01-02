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
This is a good source: [this](https://colah.github.io/posts/2015-08-Backprop/)

So, for example our backwards function of log would be:
