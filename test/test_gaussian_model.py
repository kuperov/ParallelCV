import unittest

from jax import numpy as jnp
from scipy import stats as st
import numpy as np

from ploo import run_hmc, CVPosterior, DummyProgress, GaussianModel


class TestGaussian(unittest.TestCase):

    def test_log_lik(self):
        y = jnp.array([1.,0,-1.])
        # original (not a cv fold)
        m = GaussianModel(y, mu_loc=0., mu_scale=1., sigma_shape=2., sigma_rate=2.)
        lj = m.log_joint(cv_fold=-1, mu=0.7, sigma=1.8)
        # gamma(a, rate) == gamma(a, scale=1/rate)
        ref_lp = st.norm(0., 1.).logpdf(0.7) + st.gamma(a=2., scale=1./2.).logpdf(1.8)
        ref_ll = np.sum(st.norm(0.7, 1.8).logpdf(y))
        self.assertAlmostEqual(lj, ref_lp + ref_ll, places=5)
        
        # first cv fold (index 0)
        lj = m.log_joint(cv_fold=0, mu=0.7, sigma=1.8)
        ref_ll = np.sum(st.norm(0.7, 1.8).logpdf(y[1:3]))
        self.assertAlmostEqual(lj, ref_lp + ref_ll, places=5)

        # second cv fold (index 1)
        lj = m.log_joint(cv_fold=1, mu=0.7, sigma=1.8)
        ref_ll = np.sum(st.norm(0.7, 1.8).logpdf(y[np.array([0,2])]))
        self.assertAlmostEqual(lj, ref_lp + ref_ll, places=5)

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


if __name__ == '__main__':
    unittest.main()
