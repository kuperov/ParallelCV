"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module defines the classic radon model from posteriordb.
"""

import io
import json
import os
import zipfile
from typing import Dict

import chex
import jax.scipy.stats as st
import requests
from jax import numpy as jnp

from ploo import Model
from ploo.model import ModelParams

_DATA_DIR = os.path.join(os.path.dirname(__file__), "radon_data")
_DATA_FILE = os.path.join(_DATA_DIR, "radon_all.json")


def _download_data() -> None:
    """Download data for radon model

    :raises Exception: if data cannot be downloaded from github
    """
    url = "https://github.com/stan-dev/posteriordb/raw/master/posterior_database/data/data/radon_all.json.zip"  # noqa: B950 # pylint: disable=line-too-long
    req = requests.get(url)
    if req.status_code != 200:
        raise Exception("Failed to get data file from github")
    with zipfile.ZipFile(io.BytesIO(req.content)) as archive:
        archive.extract("radon_all.json", _DATA_DIR)
    assert os.path.isfile(_DATA_FILE)


def _load_data() -> Dict[str, chex.ArrayDevice]:
    """Load radon data from _DATA_FILE.

    If data not present, automatically downloads from Github.
    """
    if not os.path.isdir(_DATA_DIR):
        _download_data()
    with open(_DATA_FILE, "r") as data_file:
        data = json.load(data_file)
    N, J = data["N"], data["J"]
    log_radon = jnp.array(data["log_radon"])
    county_index = jnp.array(data["county_idx"])
    floor_measure = jnp.array(data["floor_measure"])
    assert county_index.shape == (N,) and jnp.max(county_index) == J
    assert floor_measure.shape == (N,)
    assert log_radon.shape == (N,)
    return {
        "N": N,
        "J": J,
        "county_index": county_index,
        "floor_measure": floor_measure,
        "log_radon": log_radon,
    }


class RadonCountyIntercept(Model):
    r"""Radon county intercept model from posterior db [1]_

    The version of the radon model we are using models a level of radon
    :math:`r_i` with a random county :math:`c_i` effect :math`\alpha_{c_i}`
    and slope :math:`\beta` for explanatory variable "floor measure" :math:`f_i`.

    .. math::

        \log(r_i) \sim \mathcal{N}\left(\alpha_{c_i} + \beta f_i, \sigma_y^2 \right)

    with priors

    .. math::

        \sigma_y \sim \mathcal{N}\left(0, 1 \right) \qquad
        \alpha \sim \mathcal{N}\left(0, 10^2 \right) \qquad
        \beta \sim \mathcal{N}\left(0, 10^2 \right)

    .. _[1]: https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/radon_county_intercept.stan
    """  # noqa: B950  # pylint: disable=line-too-long

    def __init__(self) -> None:
        """Creates radon county intercept model

        Automatically downloads data from github if not available locally.
        """
        self.name = "Radon county intercept model"
        data = _load_data()
        self.N = data["N"]
        self.J = data["J"]
        self.log_radon = data["log_radon"]
        self.county_index = data["county_index"] - 1  # zero-indexed python
        self.floor_measure = data["floor_measure"]

    def log_likelihood(self, model_params: ModelParams) -> chex.ArrayDevice:
        alpha = model_params["alpha"]
        sigma_y = model_params["sigma_y"]
        beta = model_params["beta"]
        # mu is conditional mean
        mu = alpha[self.county_index] + beta * self.floor_measure
        return st.norm.logpdf(self.log_radon, loc=mu, scale=sigma_y)

    def log_prior(self, model_params: ModelParams) -> chex.ArrayDevice:
        alpha_prior = st.norm.logpdf(model_params["alpha"], loc=0.0, scale=1.0)
        sigma_prior = st.norm.logpdf(model_params["sigma_y"], loc=0.0, scale=10.0)
        beta_prior = st.norm.logpdf(model_params["beta"], loc=0.0, scale=10.0)
        return jnp.sum(alpha_prior) + sigma_prior + beta_prior

    def log_cond_pred(
        self, model_params: ModelParams, coords: chex.ArrayDevice
    ) -> chex.ArrayDevice:
        alpha = model_params["alpha"]
        sigma_y = model_params["sigma_y"]
        beta = model_params["beta"]
        mu = alpha[self.county_index[coords]] + beta * self.floor_measure[coords]
        return st.norm.logpdf(self.log_radon[coords], loc=mu, scale=sigma_y)

    def initial_value(self) -> ModelParams:
        return {
            "sigma_y": jnp.array(1.0),
            "beta": jnp.array(0.0),
            "alpha": jnp.array([0.0] * self.J),
        }
