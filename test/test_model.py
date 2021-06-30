from ploo.model import InfParams, ModelParams
import unittest

from jax import numpy as jnp
from jax.scipy import stats as st

from ploo import CVModel, LogTransform


class _GaussianVarianceModel(CVModel):
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
        return st.norm.logpdf(
            self.y, loc=self.mean, scale=jnp.sqrt(model_params["sigma_sq"])
        )

    def log_prior(self, model_params: ModelParams):
        return st.gamma.logpdf(
            model_params["sigma_sq"], a=self.prior_shape, scale=1.0 / self.prior_rate
        )

    def initial_value(self) -> ModelParams:
        return {"sigma_sq": 1.0}

    def log_pred(self, cv_index, model_params: ModelParams):
        sigma_sq = model_params["sigma_sq"]
        # this makes no sense statistically but works for testing
        return st.norm.logpdf(float(cv_index), loc=self.mean, scale=jnp.sqrt(sigma_sq))

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


class TestModelParam(unittest.TestCase):
    def setUp(self) -> None:
        y = jnp.array(5.0)
        self.model = _GaussianVarianceModel(
            y, mean=0.0, prior_shape=2.0, prior_rate=2.0
        )

    def test_log_transform(self):
        llik = self.model.log_likelihood(model_params={"sigma_sq": 2.5}, cv_fold=-1)
        lprior = self.model.log_prior({"sigma_sq": 2.5})
        ldet = self.model.log_det({"sigma_sq": 2.5})
        pot = self.model.cv_potential(inf_params={"sigma_sq": jnp.log(2.5)}, cv_fold=-1)
        self.assertEqual(llik + lprior, -pot - ldet)

    def test_initial_value(self):
        self.assertDictEqual(self.model.initial_value(), {"sigma_sq": 1.0})
        self.assertDictEqual(
            self.model.initial_value_unconstrained(), {"sigma_sq": 0.0}
        )

    def test_log_pred(self):
        for sig_sq in [0.5, 1.5]:
            self.assertEqual(
                self.model.log_pred(2, {"sigma_sq": sig_sq}),
                st.norm.logpdf(2.0, loc=0.0, scale=jnp.sqrt(sig_sq)),
            )


if __name__ == "__main__":
    unittest.main()
