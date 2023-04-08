
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

# // Model simplified
# data {
#   int<lower=0> N; // Number of rats
#   int<lower=0> Npts; // Number of data points
#   int<lower=0> rat[Npts]; // Lookup index -> rat
#   real x[Npts];
#   real y[Npts];
#   real xbar;
# }
# parameters {
#   real alpha[N];
#   real beta[N];

#   real mu_alpha;
#   real mu_beta;          // beta.c in original bugs model
#   real<lower=0> sigma_y;       // sigma in original bugs model
#   real<lower=0> sigma_alpha;
#   real<lower=0> sigma_beta;
# }
# model {
#   mu_alpha ~ normal(0, 100);
#   mu_beta ~ normal(0, 100);
#   // sigma_y, sigma_alpha, sigma_beta : flat
#   alpha ~ normal(mu_alpha, sigma_alpha); // vectorized
#   beta ~ normal(mu_beta, sigma_beta);  // vectorized
#   for (n in 1:Npts){
#     int irat;
#     irat = rat[n];
#     y[n] ~ normal(alpha[irat] + beta[irat] * (x[n] - xbar), sigma_y);
#   }
# }
# generated quantities {
#   real alpha0;
#   alpha0 = mu_alpha - xbar * mu_beta;
# }



class Theta(NamedTuple):
    alpha: jax.Array  # vector[N] alpha;
    beta: jax.Array  # vector[N] beta;
    mu_alpha: jax.Array  # real mu_alpha;
    mu_beta: jax.Array  # real mu_alpha;
    sigma_y: jax.Array  # real<lower=0> sigma_y;
    sigma_alpha: jax.Array  # real<lower=0> sigma_alpha;
    sigma_beta: jax.Array  # real<lower=0> sigma_y;


def get_data():
    """Load radon data from zip file.
    """
    # https://github.com/stan-dev/posteriordb/blob/master/posterior_database/data/data/rats_data.json.zip
    zdat = os.path.join(os.path.dirname(__file__), 'rats_data.json.zip')
    raw_data = {}
    with zipfile.ZipFile(zdat, "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = f.read()
                raw_data = json.loads(data.decode('utf-8'))
    data = {k: jnp.array(v) for (k, v) in raw_data.items()}
    data['y'] = data['y'] * 1.0  # kludge into fp32 or fp64
    return data


def get_model(data: Dict) -> Model:

    sigma_y_tfm = tfb.Exp()
    sigma_alpha_tfm = tfb.Exp()
    sigma_beta_tfm = tfb.Exp()

    N, rat = int(data['N']), data['rat']
    y, x, xbar = data['y'] * 1.0, data['x'], data['xbar']

    def make_initial_pos(key: jax.random.KeyArray) -> Theta:
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        theta = Theta(
            alpha=0.01*jax.random.normal(key=k1, shape=(N,)),
            beta=0.01*jax.random.normal(key=k2, shape=(N,)),
            mu_alpha=0.01*jax.random.normal(key=k3),
            mu_beta=0.01*jax.random.normal(key=k3),
            sigma_y=0.5 + 0.1*jax.random.normal(key=k4),
            sigma_alpha=0.5 + 0.1*jax.random.normal(key=k5),
            sigma_beta=0.5 + 0.1*jax.random.normal(key=k6),
        )
        return theta

    def logjoint_density(
            theta: Theta,
            fold_id: int,
            model_id: int,
            prior_only: bool = False) -> jax.Array:
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
        sigma_alpha = sigma_alpha_tfm.forward(theta.sigma_alpha)
        sigma_beta = sigma_beta_tfm.forward(theta.sigma_beta)
        sigma_y = sigma_y_tfm.forward(theta.sigma_y)
        sigma_alpha_ldj = sigma_alpha_tfm.forward_log_det_jacobian(theta.sigma_alpha)
        sigma_beta_ldj = sigma_beta_tfm.forward_log_det_jacobian(theta.sigma_beta)
        sigma_y_ldj = sigma_y_tfm.forward_log_det_jacobian(theta.sigma_y)
        ldj = sigma_alpha_ldj + sigma_beta_ldj + sigma_y_ldj
        # prior is same for all folds
        lp = (
            # sigma_y, sigma_alpha, sigma_beta : flat in original but half cauchy here
            #tfd.HalfCauchy(loc=0, scale=100).log_prob(sigma_y)
            #+ tfd.HalfCauchy(loc=0, scale=100).log_prob(sigma_alpha)
            #+ tfd.HalfCauchy(loc=0, scale=100).log_prob(sigma_beta)
            tfd.Normal(loc=0, scale=100).log_prob(theta.mu_alpha)
            + tfd.Normal(loc=0, scale=100).log_prob(theta.mu_beta)
            + tfd.Normal(loc=theta.mu_alpha, scale=sigma_alpha).log_prob(theta.alpha).sum()
            + tfd.Normal(loc=theta.mu_beta, scale=sigma_beta).log_prob(theta.beta).sum()
        )
        # log likelihood for fold
        y_hat = theta.alpha[rat] + theta.beta[rat] * (x - xbar)
        ll_contribs_A = tfd.Normal(loc=y_hat, scale=sigma_y).log_prob(y)  # model A
        ll_contribs_B = tfd.StudentT(df=3., loc=y_hat, scale=sigma_y).log_prob(y)  # model B
        ll_contribs = jnp.where(model_id == 0, ll_contribs_A, ll_contribs_B)
        fold_mask = (rat != fold_id) * 1.0
        lhood_mask = 1.0 * (not prior_only)
        ll = (fold_mask * ll_contribs).sum() * lhood_mask
        return lp + ll + ldj

    def log_pred(theta: Theta, fold_id: int, model_id: int) -> jax.Array:
        """Log predictive density for a given fold.
        
        Args:
        theta: model parameters
        fold_id: zero-based fold id for test set
        """
        sigma_y = sigma_y_tfm.forward(theta.sigma_y)
        # predictive log density for fold
        y_hat = theta.alpha[rat] + theta.beta[rat] * (x - xbar)
        ll_contribs_A = tfd.Normal(loc=y_hat, scale=sigma_y).log_prob(y)
        ll_contribs_B = tfd.StudentT(df=3, loc=y_hat, scale=sigma_y).log_prob(y)
        ll_contribs = jnp.where(model_id == 0, ll_contribs_A, ll_contribs_B)
        pred_mask = (rat == fold_id) * 1.0
        return (pred_mask * ll_contribs).sum()

    def to_constrained(theta: Theta) -> Theta:
        return Theta(
            alpha=theta.alpha,
            beta=theta.beta,
            mu_alpha=theta.mu_alpha,
            mu_beta=theta.mu_beta,
            sigma_y=sigma_y_tfm.forward(theta.sigma_y),
            sigma_alpha=sigma_alpha_tfm.forward(theta.sigma_alpha),
            sigma_beta=sigma_beta_tfm.forward(theta.sigma_beta),
        )

    return Model(
        num_folds=N,
        num_models=2,
        logjoint_density=logjoint_density,
        log_pred=log_pred,
        make_initial_pos=make_initial_pos,
        to_constrained=to_constrained
    )
