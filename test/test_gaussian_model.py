import unittest, os

from jax import numpy as jnp
from scipy import stats as st
import numpy as np
import pandas

from ploo import run_hmc, CVPosterior, DummyProgress, GaussianModel


def fixture(fname):
    return os.path.join(os.path.dirname(__file__), fname)


class TestGaussian(unittest.TestCase):
    def test_log_lik(self):
        y = jnp.array([1.0, 0, -1.0])

        # original (not a cv fold)
        m = GaussianModel(y, mu_loc=0.0, mu_scale=1.0, sigma_shape=2.0, sigma_rate=2.0)
        lj = m.log_joint(cv_fold=-1, mu=0.7, sigma=1.8)

        # NB gamma(a, rate) == gamma(a, scale=1/rate)
        ref_lp = st.norm(0.0, 1.0).logpdf(0.7) + st.gamma(
            a=2.0, scale=1.0 / 2.0
        ).logpdf(1.8)
        ref_ll = np.sum(st.norm(0.7, 1.8).logpdf(y))
        self.assertAlmostEqual(lj, ref_lp + ref_ll, places=5)

        # first cv fold (index 0)
        lj = m.log_joint(cv_fold=0, mu=0.7, sigma=1.8)
        ref_ll = np.sum(st.norm(0.7, 1.8).logpdf(y[1:3]))
        self.assertAlmostEqual(lj, ref_lp + ref_ll, places=5)

        # second cv fold (index 1)
        lj = m.log_joint(cv_fold=1, mu=0.7, sigma=1.8)
        ref_ll = np.sum(st.norm(0.7, 1.8).logpdf(y[np.array([0, 2])]))
        self.assertAlmostEqual(lj, ref_lp + ref_ll, places=5)

    # todo: test parameter transformation

    def test_hmc(self):
        y = GaussianModel.generate(N=200, mu=0.5, sigma=2, seed=42)
        gauss = GaussianModel(y)
        # todo: pass in canned warmup
        post = run_hmc(
            gauss, draws=1000, warmup_steps=800, chains=4, seed=42, out=DummyProgress()
        )
        self.assertIsInstance(post, CVPosterior)
        self.assertEqual(post.seed, 42)
        self.assertIs(gauss, post.model)
        p0 = next(iter(gauss.parameters))
        self.assertEqual(post.post_draws.position[p0].shape, (1000, 4))

        # Because Dan and Lauren like hypothesis tests so much
        mu_draws = post.post_draws.position["mu"].reshape((-1,))
        sigma_draws = post.post_draws.position["mu"].reshape(
            -1,
        )
        stan_post = pandas.read_csv(fixture("gaussian_post.csv"))
        ks_mu = st.kstest(stan_post["mu"], mu_draws)
        ks_sigma = st.kstest(stan_post["sigma"], sigma_draws)
        alpha = 1e-3  # rate at which our unit tests randomly fail
        self.assertGreater(ks_mu.pvalue, alpha / 2)
        self.assertGreater(ks_sigma.pvalue, alpha / 2)


if __name__ == "__main__":
    unittest.main()
