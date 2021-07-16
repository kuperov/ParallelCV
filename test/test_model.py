"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import unittest
from types import FunctionType
from typing import Tuple

import arviz as az
import chex
import distrax
import jax
from jax import numpy as jnp
from jax.scipy import stats as st

from ploo import Model, compare
from ploo.model import InfParams, ModelParams


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
        self.sigma_sq_t = distrax.Lambda(jnp.log)

    def log_prior_likelihood(self, model_params: ModelParams):
        sigma = jnp.sqrt(model_params["sigma_sq"])
        lprior = st.gamma.logpdf(
            model_params["sigma_sq"], a=self.prior_shape, scale=1.0 / self.prior_rate
        )
        llik = st.norm.logpdf(self.y, loc=self.mean, scale=sigma)
        return lprior, llik

    def initial_value(self) -> ModelParams:
        return {"sigma_sq": 1.0}

    def log_cond_pred(self, model_params: ModelParams, coords: jnp.DeviceArray):
        sigma_sq = model_params["sigma_sq"]
        lpred = st.norm.logpdf(self.y[coords], loc=self.mean, scale=jnp.sqrt(sigma_sq))
        return jnp.sum(lpred)

    def forward_transform(self, model_params: ModelParams) -> InfParams:
        return {"sigma_sq": self.sigma_sq_t.forward(model_params["sigma_sq"])}

    def inverse_transform_log_det(
        self, inf_params: InfParams
    ) -> Tuple[ModelParams, chex.ArrayDevice]:
        sst, jac = self.sigma_sq_t.inverse_and_log_det(inf_params["sigma_sq"])
        constrained = {"sigma_sq": sst}
        return constrained, jac

    @classmethod
    def generate(
        cls, N: int, mean: float, sigma_sq: float, rng_key: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Simulate the model with the given parameters

        :param N: number of observations
        :param mean: mean of process
        :param sigma_sq: true variance of generated observations
        :param rng_key: random number generator state
        :return: jax array of observations
        """
        return mean + jnp.sqrt(sigma_sq) * jax.random.normal(shape=(N,), key=rng_key)


class TestModelInferenceAndParams(unittest.TestCase):
    """Model inference and parameter manipulation"""

    def setUp(self) -> None:
        self.y = jnp.array([5.0])
        self.model = _GaussianVarianceModel(
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


class TestComparisons(unittest.TestCase):
    """Does model selection via cross-validation work?"""

    def test_compare_elpd(self):
        """Check cross-validation for model selection

        All in one big test so we only have to run one set of cross-validations.
        We aren't retaining draws here, just using the accumulated elpd.
        """
        gen_key = jax.random.PRNGKey(seed=42)
        y = _GaussianVarianceModel.generate(N=50, mean=0, sigma_sq=10, rng_key=gen_key)
        model_1 = _GaussianVarianceModel(y, mean=0.0)  # good
        model_2 = _GaussianVarianceModel(y, mean=-10.0)  # bad
        model_3 = _GaussianVarianceModel(y, mean=50.0)  # awful
        chex.clear_trace_counter()
        post_1 = model_1.inference(draws=1e3, chains=4)
        chex.clear_trace_counter()
        post_2 = model_2.inference(draws=1e3, chains=4)
        chex.clear_trace_counter()
        post_3 = model_3.inference(draws=1e3, chains=4)
        chex.clear_trace_counter()
        cv_1 = post_1.cross_validate()
        chex.clear_trace_counter()
        cv_2 = post_2.cross_validate()
        chex.clear_trace_counter()
        cv_3 = post_3.cross_validate()
        # check comparisons across CVs
        cmp_res = compare(cv_1, cv_2, cv_3)
        self.assertEqual(cmp_res.names(), ["model0", "model1", "model2"])
        for model in ["LOO", "model0", "model1", "model2"]:
            self.assertIn(model, repr(cmp_res))
        cmp_res = compare(cv_1, bad_model=cv_2, awful_model=cv_3)
        self.assertEqual(cmp_res.names(), ["model0", "bad_model", "awful_model"])
        for model in ["LOO", "model0", "bad_model", "awful_model"]:
            self.assertIn(model, repr(cmp_res))
        self.assertIs(cmp_res[0], cv_1)
        self.assertIs(cmp_res["model0"], cv_1)
        # can a cv with no draws be represented as string?
        cv1_repr = repr(cv_1)
        self.assertIsInstance(cv1_repr, str)

    def test_one_cv(self):
        """Check a single cross-validation object, retaining draws"""
        gen_key = jax.random.PRNGKey(seed=42)
        y = _GaussianVarianceModel.generate(N=50, mean=0, sigma_sq=10, rng_key=gen_key)
        model_1 = _GaussianVarianceModel(y, mean=0.0)
        post_1 = model_1.inference(draws=1e3, chains=4)
        # LOO
        cv_1 = post_1.cross_validate(thin=2)
        # 50 folds x 4 chains x 1e3/2 = 500 draws per chain
        self.assertEqual(cv_1.states["sigma_sq"].shape, (50, 4, 500))
        m1_av_f0 = cv_1.arviz(cv_fold=0)
        summ0 = az.summary(m1_av_f0)
        self.assertEqual(len(summ0), 1)  # should have 1 variable, sigma_sq
        self.assertIsInstance(m1_av_f0, az.data.inference_data.InferenceData)
        m1_av_f1 = cv_1.arviz(cv_fold=1)
        self.assertIsInstance(m1_av_f1, az.data.inference_data.InferenceData)
        cv_repr = repr(cv_1)
        self.assertIsInstance(cv_repr, str)
        # K-fold
        cv_2 = post_1.cross_validate(thin=2, scheme="KFold", k=5)
        self.assertEqual(cv_2.states["sigma_sq"].shape, (5, 4, 500))


if __name__ == "__main__":
    unittest.main()
