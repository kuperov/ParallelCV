from typing import NamedTuple
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from pcv.inference import run_cv_sel
from pcv.welford import *
from pcv.plots import plot_model_results, plot_fold_results
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

class Theta(NamedTuple):
    beta: jax.Array
    sigsq: jax.Array


def generate(
            key: jax.random.KeyArray,
            N = 100,
            beta0 = jnp.array([1.0, 1.0, 1.0, 0.5]),
            sigsq0 = jnp.array(2.0)
        ):
    y_key, X_key = jax.random.split(key)
    p = len(beta0)
    X = tfd.Normal(loc=0, scale=1).sample(sample_shape=(N, p), seed=X_key)
    y = X@beta0 + tfd.Normal(loc=0, scale=jnp.sqrt(sigsq0)).sample(sample_shape=(N,), seed=y_key)
    return y, X


def get_model(y, X, K=5):
    N, p = X.shape
    # use exp to transform sigsq to unconstrained space
    sigsq_t = tfb.Exp()

    beta_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(p), scale_diag=jnp.ones(p))
    sigsq_prior = tfd.Gamma(concentration=1.0, rate=1.0)

    def logjoint_density(theta: Theta, fold_id: int, model_id: int) -> jax.Array:
        """Log joint density for a given fold.
        
        Args:
        theta: model parameters
        fold_id: zero-based fold id for training set
        model_id: 0 for model A, 1 for model B
        """
        # transform to constrained space
        sigsq = sigsq_t.forward(theta.sigsq)
        sigsq_ldj = sigsq_t.forward_log_det_jacobian(theta.sigsq)
        # prior is same for all folds
        lp = beta_prior.log_prob(theta.beta) + sigsq_prior.log_prob(sigsq)
        # likelihood for fold
        ll_mask = ((jnp.arange(N) % K) != fold_id).astype(jnp.float32)
        # model A has all the predictors, model B is missing the last predictor
        beta_mask = jnp.where(model_id == 0, jnp.ones(p), jnp.concatenate([jnp.ones(p-1), jnp.zeros(1)]))
        ll_contribs = tfd.Normal(loc=X @ (theta.beta * beta_mask), scale=jnp.sqrt(sigsq)).log_prob(y)
        ll = (ll_mask * ll_contribs).sum()
        return lp + ll + sigsq_ldj

    # predictive density log p(y_train|theta)
    def log_pred(theta, fold_id, model_id):
        # transform to constrained space
        sigsq = sigsq_t.forward(theta.sigsq)
        pred_mask = (jnp.arange(N) % K) == fold_id
        beta_mask = jnp.where(model_id == 0, jnp.ones(p), jnp.concatenate([jnp.ones(p-1), jnp.zeros(1)]))
        ll_contribs = tfd.Normal(loc=X @ (theta.beta * beta_mask), scale=jnp.sqrt(sigsq)).log_prob(y)
        return (pred_mask * ll_contribs).sum()

    # random initialization in the constrained parameter space
    def make_initial_pos(key):
        k1, k2 = jax.random.split(key)
        theta = Theta(
        beta=jax.random.normal(key=k1, shape=(p,)),
        sigsq=jax.random.normal(key=k2))
        return theta

    return logjoint_density, log_pred, make_initial_pos
