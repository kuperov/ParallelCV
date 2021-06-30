from ploo.model import InfParams, ModelParams
from ploo.transforms import LogTransform
from jax import random, numpy as jnp
import jax.scipy.stats as st

import ploo


class GaussianModel(ploo.CVModel):
    r"""Simple Gaussian experiment for testing inference

    The model is given by
    $$ y_t \sim \mathcal{N}\left(\mu, \sigma\right), \qquad t=0,\dots,(N-1) $$

    We assume we have priors
    $$ \mu \sim N(0,1), \qquad \sigma \sim Gamma(2,2).$$
    """

    name = "Gaussian model"
    sigma_transform = LogTransform()

    def __init__(
        self,
        y: jnp.DeviceArray,
        mu_loc: float = 0.0,
        mu_scale: float = 1.0,
        sigma_shape: float = 2.0,
        sigma_rate: float = 2.0,
    ) -> None:
        self.y = y
        self.folds = jnp.arange(0, len(y))
        self.mu_loc = mu_loc
        self.mu_scale = mu_scale
        self.sigma_shape = sigma_shape
        self.sigma_rate = sigma_rate

    def log_likelihood(self, model_params: ModelParams, cv_fold=-1) -> jnp.DeviceArray:
        lik_contribs = st.norm.logpdf(
            self.y, loc=model_params["mu"], scale=model_params["sigma"]
        )
        log_lik = jnp.where(self.folds != cv_fold, lik_contribs, 0).sum()
        return log_lik

    def log_prior(self, model_params: ModelParams) -> jnp.DeviceArray:
        mu_prior = st.norm.logpdf(
            model_params["mu"], loc=self.mu_loc, scale=self.mu_scale
        )
        sigma_prior = st.gamma.logpdf(
            model_params["sigma"], a=self.sigma_shape, scale=1.0 / self.sigma_rate
        )
        return mu_prior + sigma_prior

    def initial_value(self) -> ModelParams:
        return {"mu": 0.0, "sigma": 1.0}

    def cv_folds(self):
        return len(self.y)

    def to_model_params(self, inf_params: InfParams) -> ModelParams:
        model_params = {
            "mu": inf_params["mu"],
            "sigma": self.sigma_transform.to_constrained(inf_params["sigma"]),
        }
        return model_params

    def to_inference_params(self, model_params: ModelParams) -> InfParams:
        inf_params = {
            "mu": model_params["mu"],
            "sigma": self.sigma_transform.to_unconstrained(model_params["sigma"]),
        }
        return inf_params

    def log_det(self, model_params: ModelParams) -> jnp.DeviceArray:
        return self.sigma_transform.log_det(model_params["sigma"])

    @classmethod
    def generate(cls, N=200, mu=0.5, sigma=2.0, seed=42):
        key = random.PRNGKey(seed=seed)
        return mu + sigma * random.normal(key, shape=(N,))
