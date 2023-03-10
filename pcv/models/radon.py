
import json  
import zipfile  
import os
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from typing import NamedTuple, Dict, Callable, Tuple

tfd = tfp.distributions
tfb = tfp.bijectors


# Posterior DB model
# https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/radon_hierarchical_intercept_noncentered.stan
# parameters {
#   vector[J] alpha_raw;
#   vector[2] beta;
#   real mu_alpha;
#   real<lower=0> sigma_alpha;
#   real<lower=0> sigma_y;
# }

# transformed parameters {
#   vector[J] alpha;
#   // implies: alpha ~ normal(mu_alpha, sigma_alpha);
#   alpha = mu_alpha + sigma_alpha * alpha_raw;
# }

# model {
#   vector[N] mu;
#   vector[N] muj;

#   sigma_alpha ~ normal(0, 1);
#   sigma_y ~ normal(0, 1);
#   mu_alpha ~ normal(0, 10);
#   beta ~ normal(0, 10);
#   alpha_raw ~ normal(0, 1);

#   for(n in 1:N){
#     muj[n] = alpha[county_idx[n]] + log_uppm[n] * beta[1];
#     mu[n] = muj[n] + floor_measure[n] * beta[2];
#     target += normal_lpdf(log_radon[n] | mu[n], sigma_y);
#   }
# }


class Theta(NamedTuple):
    alpha_raw: jax.Array  # vector[J] a;
    beta: jax.Array  # vector[2] beta;
    mu_alpha: jax.Array  # real mu_a;
    sigma_alpha: jax.Array  # real<lower=0> sigma_alpha;
    sigma_y: jax.Array  # real<lower=0> sigma_y;


def get_data():
    """Load radon data from zip file.
    """
    # https://github.com/stan-dev/posteriordb/blob/master/posterior_database/data/data/radon_all.json.zip
    zdat = os.path.join(os.path.dirname(__file__), 'radon_all.json.zip')
    raw_data = {}
    with zipfile.ZipFile(zdat, "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = f.read()
                raw_data = json.loads(data.decode('utf-8'))
    return {k: jnp.array(v) for (k, v) in raw_data.items()}


def get_model(data: Dict) -> Tuple[Callable, Callable, Callable]:

    sigma_a_tfm = tfb.Exp()
    sigma_y_tfm = tfb.Exp()

    J = data['J']
    county_idx = data['county_idx'] - 1
    y, log_uppm, floor_measure = data['log_radon'], data['log_uppm'], data['floor_measure']

    def make_initial_pos(key: jax.random.KeyArray) -> Theta:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        theta = Theta(
            alpha_raw=jax.random.normal(key=k1, shape=(J,)),
            beta=jax.random.normal(key=k2, shape=(2,)),
            mu_alpha=jax.random.normal(key=k3),
            sigma_alpha=jax.random.normal(key=k4),
            sigma_y=jax.random.normal(key=k5),
        )
        return theta

    def logjoint_density(theta: Theta, fold_id: int, model_id: int, prior_only: bool = False) -> jax.Array:
        """Log joint density for a given fold.
        
        Args:
            theta: model parameters
            fold_id: zero-based fold id for training set
            model_id: 0 for model A, 1 for model B
            prior_only: if True, only return prior density
        
        Returns:
            log density
        """
        # transform to constrained space
        sigma_alpha = sigma_a_tfm.forward(theta.sigma_alpha)
        sigma_y = sigma_y_tfm.forward(theta.sigma_y)
        sigma_alpha_ldj = sigma_a_tfm.forward_log_det_jacobian(theta.sigma_alpha)
        sigma_y_ldj = sigma_y_tfm.forward_log_det_jacobian(theta.sigma_y)
        ldj = sigma_y_ldj + sigma_alpha_ldj
        # prior is same for all folds
        lp = (
            tfd.Normal(loc=0, scale=0.5).log_prob(sigma_alpha)
            + tfd.Normal(loc=0, scale=0.5).log_prob(sigma_y)
            + tfd.Normal(loc=0, scale=2).log_prob(theta.mu_alpha)
            + tfd.Normal(loc=0, scale=2).log_prob(theta.beta).sum()
            + tfd.Normal(loc=0, scale=0.5).log_prob(theta.alpha_raw).sum()
        )
        # noncentering transform: alpha ~ normal(mu_alpha, sigma_alpha)
        alpha = theta.mu_alpha + sigma_alpha * theta.alpha_raw
        # likelihood for fold
        include_log_uppm = 1.0 * (model_id == 0)  # only include log_uppm in model A
        muj = alpha[county_idx] + log_uppm * theta.beta[0] * include_log_uppm
        mu = muj + floor_measure * theta.beta[1]
        ll_contribs = tfd.Normal(loc=mu, scale=sigma_y).log_prob(y)
        fold_mask = (county_idx != fold_id).astype(jnp.float32)
        ll = (fold_mask * ll_contribs).sum() * (not prior_only)
        return lp + ll + ldj

    def log_pred(theta: Theta, fold_id: int, model_id: int) -> jax.Array:
        """Log predictive density for a given fold.
        
        Args:
        theta: model parameters
        fold_id: zero-based fold id for test set
        """
        sigma_alpha = sigma_a_tfm.forward(theta.sigma_alpha)
        sigma_y = sigma_y_tfm.forward(theta.sigma_y)
        # noncentering transform: alpha ~ normal(mu_alpha, sigma_alpha)
        alpha = theta.mu_alpha + sigma_alpha * theta.alpha_raw
        # likelihood for fold
        include_log_uppm = 1.0 * (model_id == 0)  # only include log_uppm in model A
        muj = alpha[county_idx] + log_uppm * theta.beta[0] * include_log_uppm
        mu = muj + floor_measure * theta.beta[1]
        ll_contribs = tfd.Normal(loc=mu, scale=sigma_y).log_prob(y)
        fold_mask = (county_idx == fold_id).astype(jnp.float32)
        lpred = (fold_mask * ll_contribs).sum()
        return lpred

    return make_initial_pos, log_pred, logjoint_density
