import unittest
import jax
import jax.numpy as jnp

from pcv.models import rats


class TestRats(unittest.TestCase):

    def testLL(self):
        data = rats.get_data()
        model = rats.get_model(data)
        for i in range(10):
            theta = model.make_initial_pos(jax.random.PRNGKey(i))
            ll0 = model.logjoint_density(theta, -1, 0, prior_only=False)
            ll1 = model.logjoint_density(theta, -1, 1, prior_only=False)
            ll0p = model.logjoint_density(theta, -1, 0, prior_only=True)
            ll1p = model.logjoint_density(theta, -1, 1, prior_only=True)
            self.assertTrue(all(jnp.array([ll0, ll1, ll0p, ll1p]) < 0.))

    def testLLFold(self):
        data = rats.get_data()
        model = rats.get_model(data)
        for i in range(10):
            theta = model.make_initial_pos(jax.random.PRNGKey(i))
            for fold in [0, 5, 32, 78]:
                ll0 = model.logjoint_density(theta, fold, 0, prior_only=False)
                ll1 = model.logjoint_density(theta, fold, 1, prior_only=False)
                ll0p = model.logjoint_density(theta, fold, 0, prior_only=True)
                ll1p = model.logjoint_density(theta, fold, 1, prior_only=True)
                self.assertTrue(all(jnp.array([ll0, ll1, ll0p, ll1p]) < 0.))
