import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from typing import Callable

tfd = tfp.distributions


def make_positive_rule(num_folds: int, level=0.95) -> Callable:
    qtile = 1. - 0.5*(1-level)
    pos_tcrit = tfd.StudentT(df=num_folds+2, loc=0., scale=1.).quantile(qtile)
    def rule(elpd_diff, diff_cvse, model_mcse, model_ess, num_folds, num_samples, model_rhats):
        prereq = (jnp.max(model_rhats) < 1.05) and jnp.all(model_ess > 100*num_folds)
        pos = jnp.abs(elpd_diff/jnp.sqrt(jnp.sum(model_mcse**2) + diff_cvse**2)) > pos_tcrit
        return prereq and pos
    return rule


def make_positive_negative_rule(num_folds: int, level=0.95) -> Callable:
    qtile = 1. - 0.5*(1-level)
    pos_tcrit = tfd.StudentT(df=num_folds+2, loc=0., scale=1.).quantile(qtile)
    neg_tcrit = tfd.StudentT(df=num_folds, loc=0., scale=1.).quantile(qtile)
    def rule(elpd_diff, diff_cvse, model_mcse, model_ess, num_folds, num_samples, model_rhats):
        prereq = (jnp.max(model_rhats) < 1.05) and jnp.all(model_ess > 100*num_folds)
        pos = jnp.abs(elpd_diff/jnp.sqrt(jnp.sum(model_mcse**2) + diff_cvse**2)) > pos_tcrit
        neg = jnp.abs(elpd_diff/diff_cvse) < neg_tcrit
        return prereq and (pos or neg)
    return rule
