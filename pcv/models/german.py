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


# script to nab the data
def download_data():
    import pandas as pd
    import numpy as np
    data = pd.read_table(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric",
        header=None,
        delim_whitespace=True
    )
    y = -1 * (data.iloc[:, -1].values - 2)
    X = (
        data.iloc[:, :-1]
        .apply(lambda x: -1 + (x - x.min()) * 2 / (x.max() - x.min()), axis=0)
        .values
    )
    X = np.concatenate([np.ones((1000, 1)), X], axis=1)
    with open('german.json', 'w') as f:
        json.dump({'y': y.tolist(), 'X': X.tolist()}, f)


class Theta(NamedTuple):
    beta: jax.Array
    tau: jax.Array
    lmbda: jax.Array


def get_data():
    """Load german credit data from zip file.
    """
    zdat = os.path.join(os.path.dirname(__file__), 'german.json.zip')
    raw_data = {}
    with zipfile.ZipFile(zdat, "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = f.read()
                raw_data = json.loads(data.decode('utf-8'))
    return {k: jnp.array(v) for (k, v) in raw_data.items()}


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
