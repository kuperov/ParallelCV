"""diag is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
from typing import Tuple

import chex
import distrax
import jax
import jax.scipy.stats as st
from jax import numpy as jnp
from jax import random

import diag
from diag.model import CVFold, InfParams, Model, ModelParams


class GaussianModel(diag.Model):
    r"""Simple Gaussian experiment for testing inference

    The model is given by
    $$ y_t \sim \mathcal{N}\left(\mu, \sigma\right), \qquad t=0,\dots,(N-1) $$

    We assume we have priors
    $$ \mu \sim N(0,1), \qquad \sigma \sim Gamma(2,2).$$
    """

    name = "Gaussian model"
    sigma_transform = distrax.Lambda(jnp.log)

    def __init__(
        self,
        y: chex.ArrayDevice,
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

    def log_prior_likelihood(
        self, model_params: ModelParams
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        mu_prior = st.norm.logpdf(
            model_params["mu"], loc=self.mu_loc, scale=self.mu_scale
        )
        sigma_prior = st.gamma.logpdf(
            model_params["sigma"], a=self.sigma_shape, scale=1.0 / self.sigma_rate
        )
        lprior = mu_prior + sigma_prior
        log_lik = st.norm.logpdf(
            self.y, loc=model_params["mu"], scale=model_params["sigma"]
        )
        return lprior, log_lik

    def log_cond_pred(self, model_params: ModelParams, coords: CVFold):
        y_tilde = self.y[coords]  # in this example cv_fold just indexes the data
        return st.norm.logpdf(
            y_tilde, loc=model_params["mu"], scale=model_params["sigma"]
        )

    def initial_value(self) -> ModelParams:
        return {"mu": 0.0, "sigma": 1.0}

    def inverse_transform_log_det(
        self, inf_params: InfParams
    ) -> Tuple[ModelParams, chex.ArrayDevice]:
        sigma = inf_params["sigma"]
        sigma_t, sigma_ldet = self.sigma_transform.inverse_and_log_det(sigma)
        model_params = {"mu": inf_params["mu"], "sigma": sigma_t}
        return model_params, sigma_ldet

    def forward_transform(self, model_params: ModelParams) -> InfParams:
        sigma_t = self.sigma_transform.forward(model_params["sigma"])
        inf_params = {"mu": model_params["mu"], "sigma": sigma_t}
        return inf_params

    @classmethod
    def generate(cls, N=200, mu=0.5, sigma=2.0, seed=42):
        key = random.PRNGKey(seed=seed)
        return mu + sigma * random.normal(key, shape=(N,))


class GaussianVarianceModel(Model):
    r"""Test model: Gaussian with unknown variance and a single obs

    The prior for :math:`\sigma^2` is Gamma(a, b).

    Because :math:`\sigma^2` has only positive support, we need to transform
    it to cover the real line. One way is with the logarithmic transform.
    """
    name = "Gaussian variance model"

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
