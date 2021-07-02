from ploo.util import DummyProgress
from ploo.model import CVFold, InfParams, ModelParams
import unittest

import jax
from jax import numpy as jnp
from jax.scipy import stats as st

from ploo import Model, LogTransform, compare


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

    def log_likelihood(self, cv_fold, model_params: ModelParams):
        return jnp.sum(
            st.norm.logpdf(
                self.y, loc=self.mean, scale=jnp.sqrt(model_params["sigma_sq"])
            )
        )

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

    def log_cond_pred(self, model_params: ModelParams, cv_fold: CVFold):
        sigma_sq = model_params["sigma_sq"]
        return st.norm.logpdf(self.y[cv_fold], loc=self.mean, scale=jnp.sqrt(sigma_sq))

    def to_inference_params(self, model_params: ModelParams) -> InfParams:
        unconstrained = {"sigma_sq": self.sigma_sq_transform(model_params["sigma_sq"])}
        return unconstrained

    def to_model_params(self, inf_params: InfParams) -> ModelParams:
        constrained = {
            "sigma_sq": self.sigma_sq_transform.to_constrained(inf_params["sigma_sq"])
        }
        return constrained

    def log_det(self, model_params: ModelParams) -> jnp.DeviceArray:
        return self.sigma_sq_transform.log_det(model_params["sigma_sq"])

    def cv_folds(self):
        return len(self.y)  # will yield nonsense CV values of course


class TestModelParam(unittest.TestCase):
    def setUp(self) -> None:
        y = jnp.array([5.0])
        self.model = _GaussianVarianceModel(
            y, mean=0.0, prior_shape=2.0, prior_rate=2.0
        )

    def test_log_transform(self):
        llik = self.model.log_likelihood(model_params={"sigma_sq": 2.5}, cv_fold=-1)
        lprior = self.model.log_prior({"sigma_sq": 2.5})
        ldet = self.model.log_det({"sigma_sq": 2.5})
        pot = self.model.cv_potential(inf_params={"sigma_sq": jnp.log(2.5)}, cv_fold=-1)
        self.assertAlmostEqual(llik + lprior, -pot - ldet, places=5)

    def test_initial_value(self):
        self.assertDictEqual(self.model.initial_value(), {"sigma_sq": 1.0})
        self.assertDictEqual(
            self.model.initial_value_unconstrained(), {"sigma_sq": 0.0}
        )

    def test_log_pred(self):
        # fixme: this is a stupid test
        for sig_sq in [0.5, 1.5]:
            self.assertEqual(
                self.model.log_cond_pred({"sigma_sq": sig_sq}, 2),
                st.norm.logpdf(self.model.y[2], loc=0.0, scale=jnp.sqrt(sig_sq)),
            )

    def test_inference(self):
        post = self.model.inference(draws=1000, chains=4, out=DummyProgress())
        cv = post.cross_validate()
        mu_means = jnp.mean(cv.states.position["sigma_sq"])
        self.assertIsNotNone(mu_means)


class TestComparisons(unittest.TestCase):
    def test_compare_elpd(self):
        gen_key = jax.random.PRNGKey(seed=42)
        y = _GaussianVarianceModel.generate(N=50, mean=0, sigma_sq=10, rng_key=gen_key)
        m1 = _GaussianVarianceModel(y, mean=0.0)  # good
        m2 = _GaussianVarianceModel(y, mean=-10.0)  # bad
        m3 = _GaussianVarianceModel(y, mean=50.0)  # awful
        m1_post = m1.inference(out=DummyProgress())
        m2_post = m2.inference(out=DummyProgress())
        m3_post = m3.inference(out=DummyProgress())
        m1_cv = m1_post.cross_validate()
        m2_cv = m2_post.cross_validate()
        m3_cv = m3_post.cross_validate()
        cmp_res = compare(m1_cv, m2_cv, m3_cv)
        self.assertEqual(cmp_res.names(), ["model0", "model1", "model2"])
        for m in ["LOO", "model0", "model1", "model2"]:
            self.assertIn(m, repr(cmp_res))
        cmp_res = compare(m1_cv, bad_model=m2_cv, awful_model=m3_cv)
        self.assertEqual(cmp_res.names(), ["model0", "bad_model", "awful_model"])
        for m in ["LOO", "model0", "bad_model", "awful_model"]:
            self.assertIn(m, repr(cmp_res))


if __name__ == "__main__":
    unittest.main()
