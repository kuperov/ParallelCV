"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

Design information
------------------

Class diagram, like it's 2005::

    ┌─────────────────────────────┐           ┌─────────────────────┐
    │ Model                       │           │ arviz.InferenceData │
    ├─────────────────────────────┤           └───────────▲─────────┘
    │                             │                       │inherits
    │ + potential()               │instantiate┌───────────┴─────────┐
    │ + inference()───────────────┼──────────►│ _Posterior          │
    │                             │           ├─────────────────────┤
    │ Abstract methods            │           │ + post_draws        │
    │                             │        ┌──┼─+ cross_validate()  │
    │ # log_likelihood()          │        │  │ + summarize()       │
    │ # log_prior()               │        │  │ ...Arviz methods... │
    │ # cond_log_pred()           │        │  └─────────────────────┘
    │ # initial_param()           │        │instantiate
    │ # to_unconstrained_params() │        │  ┌─────────────────────┐
    │ # to_constrained_params()   │        └─►│ CrossValidation     │
    │                             │           ├─────────────────────┤
    └─────────────────────────────┘           │ + scheme            │
                                              │ + elpd              │
                            ┌─────────────────┤ + elpd_se           │
                            │ aggregate       │ + trace_plots()     │
                            │                 │ + densities()       │
                            │                 └─────────────────────┘
        ┌───────────────────┴───┐
        │ CrossValidationScheme │
        ├───────────────────────┤
        │ + mask_for()          │
        │ + coordinates_for()   │
        │ + pred_index_array()  │
        │ + mask_array()        │
        │ + summary_array()     │
        │                       │
        └▲──────▲────────▲─────▲┘
         │      │inherit │     │
         │      │        │     │
    ┌────┴┐ ┌───┴─┐ ┌────┴──┐ ┌┴────┐
    │ LOO │ │ LFO │ │ KFold │ │ LGO │
    └─────┘ └─────┘ └───────┘ └─────┘

"""

# flake8: noqa
from .compare import compare
from .cv import LFO, LGO, LOO, KFold
from .hmc import WarmupResults, warmup
from .model import Model
from .transforms import IntervalTransform, LogTransform
