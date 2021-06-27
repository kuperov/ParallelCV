import unittest

from jax import numpy as jnp

from ploo import run_hmc, CVPosterior, DummyProgress, GaussianModel


class TestGaussian(unittest.TestCase):

    def test_hmc(self):
        y = GaussianModel.generate(N=200, mu=0.5, sigma=2, seed=42)
        gauss = GaussianModel(y)
        post = run_hmc(gauss, draws=250, warmup_steps=200, chains=4, seed=42, out=DummyProgress())
        self.assertIsInstance(post, CVPosterior)
        self.assertEqual(post.seed, 42)
        self.assertIs(gauss, post.model)
        p0 = next(iter(gauss.parameters))
        self.assertEqual(post.post_draws.position[p0].shape, (250,4))
        #self.assertTrue(jnp.allclose(jnp.mean(y), jnp.mean(post.post_draws), atol=0.1))
