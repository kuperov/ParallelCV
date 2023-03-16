
import json  
import zipfile  
import os
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from typing import NamedTuple, Dict, Callable, Tuple
from pcv.model import Model


tfd = tfp.distributions
tfb = tfp.bijectors


# data {
#   int<lower=0> N;
#   int<lower=0> n_age;
#   int<lower=0> n_age_edu;
#   int<lower=0> n_edu;
#   int<lower=0> n_region_full;
#   int<lower=0> n_state;
#   int<lower=0,upper=n_age> age[N];
#   int<lower=0,upper=n_age_edu> age_edu[N];
#   vector<lower=0,upper=1>[N] black;
#   int<lower=0,upper=n_edu> edu[N];
#   vector<lower=0,upper=1>[N] female;
#   int<lower=0,upper=n_region_full> region_full[N];
#   int<lower=0,upper=n_state> state[N];
#   vector[N] v_prev_full;
#   int<lower=0,upper=1> y[N];
# }
# parameters {
#   vector[n_age] a;
#   vector[n_edu] b;
#   vector[n_age_edu] c;
#   vector[n_state] d;
#   vector[n_region_full] e;
#   vector[5] beta;
#   real<lower=0,upper=100> sigma_a;
#   real<lower=0,upper=100> sigma_b;
#   real<lower=0,upper=100> sigma_c;
#   real<lower=0,upper=100> sigma_d;
#   real<lower=0,upper=100> sigma_e;
# }
# transformed parameters {
#   vector[N] y_hat;

#   for (i in 1:N)
#     y_hat[i] = beta[1] + beta[2] * black[i] + beta[3] * female[i]
#                 + beta[5] * female[i] * black[i]
#                 + beta[4] * v_prev_full[i] + a[age[i]] + b[edu[i]]
#                 + c[age_edu[i]] + d[state[i]] + e[region_full[i]];
# }
# model {
#   a ~ normal (0, sigma_a);
#   b ~ normal (0, sigma_b);
#   c ~ normal (0, sigma_c);
#   d ~ normal (0, sigma_d);
#   e ~ normal (0, sigma_e);
#   beta ~ normal(0, 100);
#   y ~ bernoulli_logit(y_hat);
# }

class Theta(NamedTuple):
    beta: jax.Array
    tau: jax.Array
    lmbda: jax.Array


def get_data():
    """Load radon data from zip file.
    """
    with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), "election88.zip")) as f:
        with f.open("election88.json") as f:
            data = json.load(f)
    return {k: jnp.array(v) for (k, v) in data.items()}


def get_model(data: Dict) -> Model:

    tau_tfm = tfb.Exp()
    lmbda_tfm = tfb.Exp()
    y, X = data['y'], data['X']
    N, k = X.shape

    def make_initial_pos(prng_key: jax.random.KeyArray):
        return Theta(
            tau=jax.random.normal(prng_key),
            lmbda=jax.random.normal(prng_key, shape=(k,)),
            beta=jax.random.normal(prng_key, shape=(k,)),
        )

    def logjoint_density(theta: Theta):
        tau = tau_tfm.forward(theta.tau)
        lmbda = lmbda_tfm.forward(theta.lmbda)
        tau_ldj = tau_tfm.log_det_jacobian(theta.tau)
        lmbda_ldj = tau_tfm.log_det_jacobian(theta.tau).sum()
        ldj = tau_ldj + lmbda_ldj
        lp = (
            tfd.HalfCauchy(loc=0, scale=1).log_prob(tau)
            + tfd.HalfCauchy(loc=0, scale=1).log_prob(lmbda).sum()
            + tfd.Normal(loc=0, scale=tau * lmbda).log_prob(theta.beta).sum()
        )
        lpred = tfd.sigmoid(-X @ theta.beta)
        ll = tfd.Bernoulli(logits=lpred).log_prob(y).sum()
        return lp + ll + ldj

    def log_pred(theta: Theta):
        lpred = tfd.sigmoid(-X @ theta.beta)
        ll = tfd.Bernoulli(logits=lpred).log_prob(y).sum()
        return ll

    def to_constrained(theta: Theta):
        return Theta(
            beta=theta.beta,
            tau=tau_tfm.forward(theta.tau),
            lmbda=lmbda_tfm.forward(theta.lmbda),
        )

    return Model(
        num_folds=N,
        num_models=2,
        log_pred=log_pred,
        logjoint_density=logjoint_density,
        to_constrained=to_constrained,
        make_initial_pos=make_initial_pos,
    )
