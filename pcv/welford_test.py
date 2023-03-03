import unittest

import jax
import jax.numpy as jnp

from pcv.welford import *


class TestUnivariateWelford(unittest.TestCase):

    def testMoments(self):
        w0 = welford_init(K=500)
        key = jax.random.PRNGKey(0)
        xs = 200 + 200 * jax.random.normal(key=key, shape=(100,))
        w, _ = jax.lax.scan(lambda carry_w, x: (welford_add(x, carry_w), None), w0, xs)
        self.assertAlmostEqual(welford_mean(w), jnp.mean(xs), places=2)
        self.assertAlmostEqual(welford_var(w, ddof=0), jnp.var(xs), places=2)
        self.assertAlmostEqual(welford_var(w, ddof=1), jnp.var(xs, ddof=1), places=2)
        self.assertAlmostEqual(welford_mad(w), jnp.abs(xs - w0.K).mean(), places=2)

    def testBatchMoments(self):
        n, b = 100, 10
        key = jax.random.PRNGKey(0)
        xs = jax.random.normal(key=key, shape=(n*b,))
        xbs = xs.reshape((n, b))  # n batches of size=b
        bmeans = xbs.mean(axis=1)
        w0 = batch_welford_init(K=0.0, batch_size=b)
        w, _ = jax.lax.scan(lambda carry_w, x: (batch_welford_add(x, carry_w), None), w0, xs)
        self.assertAlmostEqual(batch_welford_mean(w), bmeans.mean(), places=2)
        self.assertAlmostEqual(batch_welford_var(w, ddof=1), bmeans.var(ddof=1), places=2)
        self.assertAlmostEqual(batch_welford_var(w, ddof=0), bmeans.var(ddof=0), places=2)


class TestLogWelford(unittest.TestCase):

    def testMoments(self):
        w0 = log_welford_init(shape=tuple())
        key = jax.random.PRNGKey(0)
        n = 5
        lxs = jax.random.normal(key=key, shape=(n,))
        xs = jnp.exp(lxs)
        w, _ = jax.lax.scan(lambda carry_w, lx: (log_welford_add(lx, carry_w), None), w0, lxs)
        mean = jax.scipy.special.logsumexp(lxs) - jnp.log(n)
        self.assertAlmostEqual(log_welford_mean(w), mean, places=2)
        self.assertAlmostEqual(log_welford_var(w), jnp.log(jnp.var(xs)), places=2)
        self.assertAlmostEqual(log_welford_var(w, ddof=0), jnp.log(jnp.var(xs, ddof=0)), places=2)
        self.assertAlmostEqual(log_welford_var(w, ddof=1), jnp.log(jnp.var(xs, ddof=1)), places=2)


class TestMultivariateWelford(unittest.TestCase):

    def testVectorMoments(self):
        key = jax.random.PRNGKey(0)
        data = jax.random.multivariate_normal(key=key, mean=jnp.array([0., 1., 2.]), cov=jnp.array([[1., 0.5, 0.], [0.5, 1., 0.], [0., 0., 1.]]), shape=(1000,))
        def update(w_carry, x):
            return (vector_welford_add(x, w_carry), None)
        w0 = vector_welford_init(K=jnp.zeros(3))
        w, _ = jax.lax.scan(update, w0, data)
        self.assertLess(jnp.linalg.norm(vector_welford_cov(w) - jnp.cov(data.T)), 1e-5)
        self.assertLess(jnp.linalg.norm(vector_welford_cov(w, ddof=0) - jnp.cov(data.T, ddof=0)), 1e-5)
        self.assertLess(jnp.linalg.norm(vector_welford_mean(w) - data.mean(axis=0)), 1e-5)

    def testBatchVectorMoments(self):
        n, b, p = 100, 10, 3
        key = jax.random.PRNGKey(0)
        data = jax.random.multivariate_normal(key=key, mean=jnp.array([0., 1., 2.]), cov=jnp.array([[1., 0.5, 0.], [0.5, 1., 0.], [0., 0., 1.]]), shape=(n*b,))
        xbs = data.reshape((n, b, p))  # n batches of size=b
        def update(w_carry, x):
            return (batch_vector_welford_add(x, w_carry), None)
        w0 = batch_vector_welford_init(K=jnp.zeros(p), batch_size=b)
        w, _ = jax.lax.scan(update, w0, data)
        bmeans = xbs.mean(axis=1)
        self.assertLess(jnp.linalg.norm(batch_vector_welford_mean(w) - bmeans.mean(axis=0)), 1e-5)
        self.assertLess(jnp.linalg.norm(batch_vector_welford_cov(w, ddof=1) - jnp.cov(bmeans.T, ddof=1)), 1e-5)
        self.assertLess(jnp.linalg.norm(batch_vector_welford_cov(w, ddof=0) - jnp.cov(bmeans.T, ddof=0)), 1e-2)
