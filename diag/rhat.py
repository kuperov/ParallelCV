from typing import Callable, Dict, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

import blackjax.adaptation as adaptation
import blackjax.mcmc as mcmc
import blackjax.sgmcmc as sgmcmc
import blackjax.smc as smc
import blackjax.vi as vi
from blackjax.base import AdaptationAlgorithm, MCMCSamplingAlgorithm, VIAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import Array, PRNGKey, PyTree
from blackjax.kernels import ghmc

import blackjax
import jax
import chex
import jax.numpy as jnp
import arviz as az
from tensorflow_probability.substrates import jax as tfp
from collections import namedtuple
import matplotlib.pyplot as plt
from typing import NamedTuple
from jax.tree_util import tree_map, tree_structure, tree_flatten, tree_unflatten
from jax.scipy.special import logsumexp
import pandas as pd


def split_rhat(means: chex.Array, vars: chex.Array, n: int) -> float:
    """Compute a single split Rhat from summary statistics of split chains.

    Args:
        means: means of split chains
        vars:  variances of split chains
        n:     number of draws per split chain (ie half draws in an original chain)
    """
    W = jnp.mean(vars, axis=1)
    #m = means.shape[1]  # number of split chains
    B = n*jnp.var(means, ddof=1, axis=1)
    varplus = (n-1)/n*W + B/n
    Rhat = jnp.sqrt(varplus/W)
    return Rhat

def split_rhat_welford(ws: WelfordState) -> float:
    """Compute split Rhat from Welford state of split chains.

    Args:
        ws: Welford state of split chains
    
    Returns:
        split Rhat: array of split Rhats
    """
    means = jax.vmap(welford_mean)(ws)
    vars = jax.vmap(welford_var)(ws)
    n = ws.n[:,0,...]  # we aggregate over chain dim, axis=1
    return split_rhat(means, vars, n)

def folded_split_rhat_welford(ws: WelfordState) -> float:
    """Compute folded split Rhat from Welford state of split chains.

    Args:
        ws: Welford state of split chains
    
    Returns:
        folded split Rhat: array of folded split Rhats
    """
    mads = jax.vmap(welford_mad)(ws)
    vars = jax.vmap(welford_var)(ws)
    n = ws.n[:,0,...]
    return split_rhat(mads, vars, n)

def rhat(welford_tree):
    """Compute split Rhat and folded split Rhat from welford states of split chains.

    This version assumes there are multiple posteriors, so that the states have dimension
    (cv_fold #, chain #, half #, ...).

    Args:
        welford_tree: pytree of Welford states for split chains
    
    Returns:
        split Rhat: pytree pytree of split Rhats
        folded split Rhat: pytree of folded split Rhats
    """
    # collapse axis 1 (chain #) and axis 2 (half #) to a single dimension
    com_chains = tree_map(lambda x: jnp.reshape(x, (x.shape[0], -1, *x.shape[3:])), welford_tree)
    sr = tree_map(split_rhat_welford, com_chains, is_leaf=lambda x: isinstance(x, WelfordState))
    fsr = tree_map(folded_split_rhat_welford, com_chains, is_leaf=lambda x: isinstance(x, WelfordState))
    return sr, fsr

def rhat_summary(fold_states):
    """Compute split Rhat and folded split Rhat from welford states of split chains.

    This version assumes there are multiple posteriors, so that the states have dimension
    (cv_fold #, chain #, half #, ...).

    Args:
        fold_states: pytree of Welford states for split chains
    
    Returns:
        pandas data frame summarizing rhats
    """
    par_rh, par_frh = rhat(fold_states.param_ws)
    pred_rh, pred_frh = rhat(fold_states.pred_ws)
    K = pred_rh.shape[0]
    rows = []
    max_row = None
    for i in range(K):
        for (par, pred, meas) in [(par_rh, pred_rh, 'Split Rhat'), (par_frh, pred_frh, 'Folded Split Rhat')]:
            row = {'fold': f'Fold {i}', 'measure': meas}
            for j, parname in enumerate(par._fields):
                if jnp.ndim(par[j]) > 1:
                    # vector parameter, add a column for each element
                    for k in range(par[j].shape[1]):
                        row[f'{parname}[{k}]'] = float(par[j][i][k])
                else:
                    row[parname] = float(par[j][i])
            row['log p'] = float(pred[i])
            rows.append(row)
            if max_row:
                max_row = {k: max_row[k] if isinstance(max_row[k], str) else max(max_row[k], row[k]) for k in max_row}
            else:
                max_row = row.copy()
                max_row.update({'fold': 'All folds', 'measure': 'Max'})
    rows.append(max_row)
    return pd.DataFrame(rows)