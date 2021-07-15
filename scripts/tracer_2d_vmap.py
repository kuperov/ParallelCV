#!./venv/bin/python3

# Script to check that 2-dimensional vmap works as expected

import jax
from jax import numpy as jnp

X = 2 * jnp.arange(99).reshape((9, 11))
Y = -jnp.arange(99).reshape((9, 11))


def f(x, y):
    assert x.ndim + y.ndim == 0
    return x + y


# vmap docs https://jax.readthedocs.io/en/latest/jax.html?highlight=vmap#jax.vmap
# use axis in the singular so I'm assuming we can't range over multiple axes
f_row = jax.vmap(f, in_axes=[0, 0])
f_mat = jax.vmap(f_row, in_axes=[1, 1])

f_mat(X, Y)
