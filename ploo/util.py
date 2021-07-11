"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import time

import chex
import jax
import numpy as np
import xarray as xr


class Timer:
    """Timer for measuring and reporting performance."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def sec(self) -> float:
        return time.perf_counter() - self.started

    def __str__(self):
        return f"{self.sec:1.01f} sec"


def print_devices():
    """Print summary of available devices to console."""
    device_list = [f"{d.device_kind} ({d.platform}{d.id})" for d in jax.devices()]
    if len(device_list) > 0:
        print(f'Detected devices: {", ".join(device_list)}')
    else:
        print("Only CPU is available. Check cuda/cudnn library versions.")


def to_posterior_dict(post_draws: chex.ArrayDevice) -> xr.Dataset:
    """Construct xarrays for ArviZ

    Converts all objects to in-memory numpy arrays. This involves a lot of copying,
    of course, but ArviZ chokes if given jax.numpy arrays.

    :param post_draws: dict of posterior draws, keyed by parameter name
    :returns: xarray dataset suitable for passing to az.InferenceData
    """
    first_param = next(iter(post_draws))
    chains = post_draws[first_param].shape[0]
    draws = post_draws[first_param].shape[1]
    post_draw_map = {}
    coords = {  # gets updated for
        "chain": (["chain"], np.arange(chains)),
        "draw": (["draw"], np.arange(draws)),
    }
    for var, drws in post_draws.items():
        # dimensions are chain, draw number, variable dim 0, variable dim 1, ...
        extra_dims = [(f"{var}{i}", length) for i, length in enumerate(drws.shape[2:])]
        keys = ["chain", "draw"] + [n for n, len in extra_dims]
        post_draw_map[var] = (keys, np.asarray(drws))
        for dimname, length in extra_dims:
            coords[dimname] = ([dimname], np.arange(length))

    posterior = xr.Dataset(post_draw_map, coords=coords)
    return posterior
