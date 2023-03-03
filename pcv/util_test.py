import unittest

import jax
import jax.numpy as jnp
import chex


class TestUtil(unittest.TestCase):

    def test_logvar(self):
        from pcv.util import logvar
        # basic 1-dimensional
        x = jnp.arange(1,6)
        logx = jnp.log(x)
        self.assertEqual(logvar(logx), jnp.log(jnp.var(x)))
        # n-dimensional
        check = lambda x, y: chex.assert_tree_all_close(x, y, atol=1e-5)
        for shape in [(10,), (4,5), (4,5,6)]:
            for axis in range(len(shape)):
                logx = jax.random.normal(jax.random.PRNGKey(0), shape)
                x = jnp.exp(logx)
                check(logvar(logx, axis=axis), jnp.log(jnp.var(x, axis=axis)))
                check(logvar(logx, axis=axis, ddof=0), jnp.log(jnp.var(x, axis=axis, ddof=0)))
                check(logvar(logx, axis=axis, ddof=1), jnp.log(jnp.var(x, axis=axis, ddof=1)))
