"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import unittest
from types import FunctionType

import chex
import jax
from arviz.data.inference_data import InferenceData
from jax import numpy as jnp
from jax.scipy import stats as st

from ploo import LogTransform, Model, compare
from ploo.model import InfParams, ModelParams
from ploo.util import DummyProgress


class _GaussianVarianceModel(Model):
    r"""Test model: Gaussian with unknown variance and a single obs

    The prior for :math:`\sigma^2` is Gamma(a, b).

    Because :math:`\sigma^2` has only positive support, we need to transform
    it to cover the real line. One way is with the logarithmic transform.
    """

    def __init__(
        self,
        y: jnp.DeviceArray,
        mean: float = 0.0,
        prior_shape: float = 2.0,
        prior_rate: float = 2.0,
    ) -> None:
        self.y = y
        self.mean = mean
        self.prior_shape = prior_shape
        self.prior_rate = prior_rate
        self.sigma_sq_transform = LogTransform()

    def log_likelihood(self, model_params: ModelParams):
        sigma = jnp.sqrt(model_params["sigma_sq"])
        return st.norm.logpdf(self.y, loc=self.mean, scale=sigma)

    def log_prior(self, model_params: ModelParams):
        return st.gamma.logpdf(
            model_params["sigma_sq"], a=self.prior_shape, scale=1.0 / self.prior_rate
        )

    @classmethod
    def generate(
        cls, N: int, mean: float, sigma_sq: float, rng_key: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        return mean + jnp.sqrt(sigma_sq) * jax.random.normal(shape=(N,), key=rng_key)

    def initial_value(self) -> ModelParams:
        return {"sigma_sq": 1.0}

    def log_cond_pred(self, model_params: ModelParams, coords: jnp.DeviceArray):
        sigma_sq = model_params["sigma_sq"]
        lpred = st.norm.logpdf(self.y[coords], loc=self.mean, scale=jnp.sqrt(sigma_sq))
        return jnp.sum(lpred)

    def to_inference_params(self, model_params: ModelParams) -> InfParams:
        unconstrained = {
            "sigma_sq": self.sigma_sq_transform.to_unconstrained(
                model_params["sigma_sq"]
            )
        }
        return unconstrained

    def to_model_params(self, inf_params: InfParams) -> ModelParams:
        constrained = {
            "sigma_sq": self.sigma_sq_transform.to_constrained(inf_params["sigma_sq"])
        }
        return constrained

    def log_det(self, model_params: ModelParams) -> jnp.DeviceArray:
        return self.sigma_sq_transform.log_det(model_params["sigma_sq"])


class TestModelParam(unittest.TestCase):
    def setUp(self) -> None:
        self.y = jnp.array([5.0])
        self.model = _GaussianVarianceModel(
            self.y, mean=0.0, prior_shape=2.0, prior_rate=2.0
        )

    def test_log_transform(self):
        llik = self.model.log_likelihood(model_params={"sigma_sq": 2.5})
        lprior = self.model.log_prior({"sigma_sq": 2.5})
        ldet = self.model.log_det({"sigma_sq": 2.5})
        pot = self.model.potential(inf_params={"sigma_sq": jnp.log(2.5)})
        self.assertAlmostEqual(llik + lprior, -pot - ldet, places=5)

    def test_initial_value(self):
        self.assertDictEqual(self.model.initial_value(), {"sigma_sq": 1.0})
        self.assertDictEqual(
            self.model.initial_value_unconstrained(), {"sigma_sq": 0.0}
        )

    def test_log_pred(self):
        for sig_sq in [0.5, 1.5]:
            self.assertEqual(
                self.model.log_cond_pred({"sigma_sq": sig_sq}, 2),
                st.norm.logpdf(self.model.y[2], loc=0.0, scale=jnp.sqrt(sig_sq)),
            )

    def test_inference(self):
        """Check full-data inference and posterior"""
        post = self.model.inference(draws=1000, chains=4, out=DummyProgress())
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


class TestComparisons(unittest.TestCase):
    """Does model selection via cross-validation work?"""

    def test_compare_elpd(self):
        """Check cross-validation for one posterior, and model selection

        All in one big test so we only have to run one set of cross-validations
        """
        gen_key = jax.random.PRNGKey(seed=42)
        y = _GaussianVarianceModel.generate(N=50, mean=0, sigma_sq=10, rng_key=gen_key)
        model_1 = _GaussianVarianceModel(y, mean=0.0)  # good
        model_2 = _GaussianVarianceModel(y, mean=-10.0)  # bad
        model_3 = _GaussianVarianceModel(y, mean=50.0)  # awful
        chex.clear_trace_counter()
        post_1 = model_1.inference(draws=1e3, chains=4, out=DummyProgress())
        chex.clear_trace_counter()
        post_2 = model_2.inference(draws=1e3, chains=4, out=DummyProgress())
        chex.clear_trace_counter()
        post_3 = model_3.inference(draws=1e3, chains=4, out=DummyProgress())
        chex.clear_trace_counter()
        cv_1 = post_1.cross_validate()
        chex.clear_trace_counter()
        cv_2 = post_2.cross_validate()
        chex.clear_trace_counter()
        cv_3 = post_3.cross_validate()
        # check just CV posterior m1
        m1_av_f0 = cv_1.arviz(cv_fold=0)
        self.assertIsInstance(m1_av_f0, InferenceData)
        m1_av_f1 = cv_1.arviz(cv_fold=1)
        self.assertIsInstance(m1_av_f1, InferenceData)
        # check comparisons across CVs
        cmp_res = compare(cv_1, cv_2, cv_3)
        self.assertEqual(cmp_res.names(), ["model0", "model1", "model2"])
        for m in ["LOO", "model0", "model1", "model2"]:
            self.assertIn(m, repr(cmp_res))
        cmp_res = compare(cv_1, bad_model=cv_2, awful_model=cv_3)
        self.assertEqual(cmp_res.names(), ["model0", "bad_model", "awful_model"])
        for m in ["LOO", "model0", "bad_model", "awful_model"]:
            self.assertIn(m, repr(cmp_res))
        self.assertIs(cmp_res[0], cv_1)
        self.assertIs(cmp_res["model0"], cv_1)


if __name__ == "__main__":
    unittest.main()
