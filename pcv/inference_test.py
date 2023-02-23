import unittest

from pcv.inference import *

from typing import NamedTuple
import chex
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from pcv.inference import fold_posterior, inference_loop, offline_inference_loop, estimate_elpd, rhat_summary
import arviz as az
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


# use exp to transform sigsq to unconstrained space
sigsq_t = tfb.Exp()

p = 4
beta_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(p), scale_diag=jnp.ones(p))
sigsq_prior = tfd.Gamma(concentration=1.0, rate=1.0)


class Theta(NamedTuple):
    beta: chex.Array
    sigsq: chex.Array


def _init_model(data_key, N = 100, p = p):
    y_key, X_key = jax.random.split(data_key)
    beta0 = jnp.arange(p)
    sigsq0 = jnp.array(2.0)
    X = tfd.Normal(loc=0, scale=1).sample(sample_shape=(N, p), seed=X_key)
    y = X@beta0 + tfd.Normal(loc=0, scale=jnp.sqrt(sigsq0)).sample(sample_shape=(N,), seed=y_key)

    def logjoint_density(theta: Theta, fold_id: int = -1) -> chex.Array:
        """Log joint density for a given fold.
        
        Args:
        theta: model parameters
        fold_id: zero-based fold id for training set, use -1 for all data.
        """
        # transform to constrained space
        sigsq = sigsq_t.forward(theta.sigsq)
        sigsq_ldj = sigsq_t.forward_log_det_jacobian(theta.sigsq)
        # prior is same for all folds
        lp = beta_prior.log_prob(theta.beta) + sigsq_prior.log_prob(sigsq)
        # likelihood for fold
        mask = 1.0 * ((jnp.arange(N) % 5) != fold_id)
        ll_contribs = tfd.Normal(loc=X@theta.beta, scale=jnp.sqrt(sigsq)).log_prob(y)
        ll = (mask * ll_contribs).sum()
        return lp + ll

    # predictive density log p(y_train|theta)
    def log_pred(theta, fold_id):
        # transform to constrained space
        sigsq = sigsq_t.forward(theta.sigsq)
        pred_mask = 1.0 * ((jnp.arange(N) % 5) == fold_id)
        npred = pred_mask.sum()
        esq = ((X @ theta.beta - y) ** 2) * pred_mask
        return -0.5 * (
            npred * jnp.log(2 * jnp.pi)
            + npred * jnp.log(sigsq)
            + esq.sum()/sigsq
        )

    # random initialization in the constrained parameter space
    def make_initial_pos(key):
        k1, k2 = jax.random.split(key)
        theta = Theta(
        beta=jax.random.normal(key=k1, shape=(p,)),
        sigsq=jax.random.normal(key=k2))
        return theta

    return logjoint_density, log_pred, make_initial_pos


class TestInference(unittest.TestCase):

    def test_inference(self):
        model_key, inference_key = jax.random.split(jax.random.PRNGKey(123))
        
        logjoint_density, log_pred, make_initial_pos = _init_model(model_key)

        def make_fold(fold_id):
            results = fold_posterior(
                prng_key=inference_key,
                inference_loop=inference_loop,
                logjoint_density=lambda theta: logjoint_density(theta, fold_id),
                log_p=lambda theta: log_pred(theta, fold_id),
                make_initial_pos=make_initial_pos,
                num_chains=10,
                num_samples=10_000,
                warmup_iter=2000)
            return results

        def replay_fold(fold_id, inference_key=inference_key):
            results, trace = fold_posterior(
                prng_key=inference_key,
                inference_loop=offline_inference_loop,
                logjoint_density=lambda theta: logjoint_density(theta, fold_id),
                log_p=lambda theta: log_pred(theta, fold_id),
                make_initial_pos=make_initial_pos,
                num_chains=10,
                num_samples=2000,
                warmup_iter=1000)
            pos = trace.position
            theta_dict = az.convert_to_inference_data(dict(beta=pos.beta, sigsq=jax.vmap(sigsq_t.forward)(pos.sigsq)))
            trace_az = az.convert_to_inference_data(theta_dict)
            return results, trace_az

        online_fold_states = jax.vmap(make_fold)(jnp.arange(5))
        offline_fold_states, offline_fold_traces = jax.vmap(replay_fold)(jnp.arange(5))
        