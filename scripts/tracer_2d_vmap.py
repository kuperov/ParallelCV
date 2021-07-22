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

result1 = f_mat(X, Y)


# ----------------------------------------------------------------------------

# over 1d arrays

W = jnp.arange(10)
Z = jnp.arange(10, 20)

# same f as before

f1 = jax.vmap(f, in_axes=[0, None])
f2 = jax.vmap(f1, in_axes=[None, 0])
result2 = f2(W, Z)

# ----------------------------------------------------------------------------

# over dicts

param = {'a': jnp.arange(10), 'b': jnp.arange(10, 20)}

def g(p):
    return p['a'] + p['b']

result3 = jax.vmap(g, 0, 0)(param)

def h(p, z):
    return g(p) + z

h1 = jax.vmap(h, in_axes=[0, None])
h2 = jax.vmap(h1, in_axes=[None, 0])
result4 = h2(param, Z)
