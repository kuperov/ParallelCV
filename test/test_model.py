"""diag is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import unittest
from types import FunctionType

from jax import numpy as jnp
from jax.scipy import stats as st

from diag.models import GaussianVarianceModel


class TestModelInferenceAndParams(unittest.TestCase):
    """Model inference and parameter manipulation"""

    def setUp(self) -> None:
        self.y = jnp.array([5.0])
        self.model = GaussianVarianceModel(
            self.y, mean=0.0, prior_shape=2.0, prior_rate=2.0
        )

    def test_log_transform(self):
        """Log transformation between parameter on half line and full line"""
        model_params = {"sigma_sq": 2.5}
        inf_params = self.model.forward_transform(model_params)
        self.assertAlmostEqual(
            jnp.log(model_params["sigma_sq"]), inf_params["sigma_sq"]
        )
        lprior, llik = self.model.log_prior_likelihood(model_params)
        _, ldet = self.model.inverse_transform_log_det(inf_params)
        pot = self.model.cv_potential(inf_params=inf_params, likelihood_mask=1.0)
        self.assertAlmostEqual(llik + lprior, -pot - ldet, places=5)

    def test_initial_value(self):
        """Initial parameter value for seeding MCMC"""
        self.assertDictEqual(self.model.initial_value(), {"sigma_sq": 1.0})
        self.assertDictEqual(self.model.initial_value_transformed(), {"sigma_sq": 0.0})

    def test_log_pred(self):
        """Log predictive, log p(ỹ | y)"""
        for sig_sq in [0.5, 1.5]:
            self.assertEqual(
                self.model.log_cond_pred({"sigma_sq": sig_sq}, 2),
                st.norm.logpdf(self.model.y[2], loc=0.0, scale=jnp.sqrt(sig_sq)),
            )

    def test_inference(self):
        """Check full-data inference and posterior"""
        post = self.model.inference(draws=1000, chains=4)
        # check delegated arviz methods are listed and actually functions
        self.assertIn("plot_density", dir(post))
        self.assertIsInstance(post.plot_density, FunctionType)
        self.assertIn("loo", dir(post))
        self.assertIsInstance(post.loo, FunctionType)
        # sensible posterior?
        sig_sq_means = jnp.mean(post.post_draws["sigma_sq"])
        self.assertIsInstance(sig_sq_means, jnp.DeviceArray)
        # smoke test summary table
        post_table = str(post)
        self.assertIn("4,000 draws", post_table)
        self.assertIn("1,000 iterations", post_table)
        self.assertIn("4 chains", post_table)
        self.assertIn("sigma_sq", post_table)


if __name__ == "__main__":
    unittest.main()
