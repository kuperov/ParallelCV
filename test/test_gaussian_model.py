"""diag is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""

import unittest
from test.util import fixture

import numpy as np
import pandas
from jax import numpy as jnp
from scipy import stats as st

from diag.model import _Posterior
from diag.models import GaussianModel


class TestGaussian(unittest.TestCase):
    """Simple tests on Gaussian model"""

    def setUp(self) -> None:
        self.y = jnp.array([1.0, 0, -1.0])
        self.model = GaussianModel(
            self.y, mu_loc=0.0, mu_scale=1.0, sigma_shape=2.0, sigma_rate=2.0
        )

    def test_log_lik(self):
        """Does the joint prior & likelihood evaluate correctly"""

        y = jnp.array([1.0, 0, -1.0])

        # full_data (not a cv fold)
        model = GaussianModel(
            y, mu_loc=0.0, mu_scale=1.0, sigma_shape=2.0, sigma_rate=2.0
        )
        param = {"mu": 0.7, "sigma": 1.8}
        log_prior, log_lhood = model.log_prior_likelihood(param)
        # NB gamma(a, rate) == gamma(a, scale=1/rate)
        ref_lp = st.norm(0.0, 1.0).logpdf(0.7) + st.gamma(
            a=2.0, scale=1.0 / 2.0
        ).logpdf(1.8)
        ref_ll = np.sum(st.norm(0.7, 1.8).logpdf(y))
        self.assertAlmostEqual(
            log_prior + jnp.sum(log_lhood), ref_lp + ref_ll, places=5
        )

    def test_transforms(self):
        """Are transformations applied correctly"""
        model_p = {"mu": 0.5, "sigma": 2.5}
        trans_p = {"mu": 0.5, "sigma": jnp.log(2.5)}
        transformed, _ = self.model.inverse_transform_log_det(trans_p)
        self.assertAlmostEqual(jnp.array(model_p["mu"]), transformed["mu"], places=5)
        self.assertAlmostEqual(
            jnp.array(model_p["sigma"]), transformed["sigma"], places=5
        )
        transformed = self.model.forward_transform(model_p)
        self.assertAlmostEqual(jnp.array(trans_p["mu"]), transformed["mu"], places=5)
        self.assertAlmostEqual(
            jnp.array(trans_p["sigma"]), transformed["sigma"], places=5
        )

    def test_hmc(self):
        """Does inference with HMC behave as expected"""

        y = GaussianModel.generate(N=200, mu=0.5, sigma=2.0, seed=42)
        gauss = GaussianModel(y)
        post = gauss.inference(
            draws=1000,
            warmup_steps=800,
            chains=4,
            seed=42,
        )
        self.assertIsInstance(post, _Posterior)
        self.assertIs(gauss, post.model)
        first_par = next(iter(gauss.parameters()))
        self.assertEqual(post.post_draws[first_par].shape, (4, 1000))

        self.assertAlmostEqual(jnp.mean(y), jnp.mean(post.post_draws["mu"]), places=1)
        self.assertAlmostEqual(jnp.std(y), jnp.mean(post.post_draws["sigma"]), places=1)

        # Because Dan and Lauren like hypothesis tests so much
        mu_draws = post.post_draws["mu"].reshape((-1,))
        sigma_draws = post.post_draws["sigma"].reshape((-1,))
        stan_post = pandas.read_csv(fixture("gaussian_post.csv"))
        ks_mu = st.kstest(stan_post["mu"], mu_draws)
        ks_sigma = st.kstest(stan_post["sigma"], sigma_draws)
        alpha = 1e-3  # rate at which our unit tests randomly fail
        self.assertGreater(ks_mu.pvalue, alpha / 2)
        self.assertGreater(ks_sigma.pvalue, alpha / 2)


if __name__ == "__main__":
    unittest.main()
