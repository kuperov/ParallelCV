"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

Design information
------------------

Class diagram::

    ┌──────────┐     ┌──────────────────────┐     ┌───────────────────────┐
    │ CVKernel │     │  CrossValidatedModel │     │ CrossValidationScheme │
    │          │     │                      │     │                       │
    │          │has-a│  * potential         │has-a│ * __iter__            │
    │          ├─────►                      ├─────►                       │
    │          │     │  * num_folds         │     │ * mask_for            │
    └──────────┘     │                      │     │                       │
                     │  * initial_param     │     │ * coordinates_for     │
                     │                      │     │                       │
                     └───────────────┬──────┘     └───────────────────────┘
                                has-a│
                     ┌───────────────▼──────┐
                     │  Model               │
                     │                      │
                     │  * log_likelihood    │
                     │                      │
                     │  * log_prior         │
                     │                      │
                     │  * initial_param     │
                     │                      │
                     └──────────────────────┘

"""

# flake8: noqa
from .compare import compare
from .cv import LFO, LOO, KFold
from .hmc import WarmupResults, warmup
from .model import Model
from .transforms import IntervalTransform, LogTransform
from .util import DummyProgress, Progress
