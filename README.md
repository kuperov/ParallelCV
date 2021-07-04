ploo: Parallel Brute Force LOO-CV
=================================

This is an experiment to try and implement LOO-CV with parallel processing on a GPU.

The project roadmap has been moved to [Github issues](https://github.com/kuperov/ploo/milestones)

Setting up
----------

You'll need python ≥ 3.7. It's probably easiest to use Anaconda but YMMV.

On a reasonable OS you should be able to create the virtual environment that contains runtime and development dependencies with:

    git clone git@github.com:kuperov/ploo.git
    cd ploo
    make config

This will add a pre-commit git hook that runs the linter. The Makefile probably won't work on Windows without WSL.

Check everything is working by running the unit tests:

    make test

Before commiting code, run the code formatter and linter (the pre-commit hook should lint automatically):

    make pretty
    make lint

Example
-------

To fit a model, you need to specify its likelihood, prior, and conditional predictive score.
You do this by providing a subclass of `ploo.Model`:

```{python}
from jax import random, numpy as jnp
import jax.scipy.stats as st
import ploo


class GaussianModel(ploo.Model):
    name = "Gaussian mean model"

    def __init__(self, y):
        self.y = y
        self.folds = jnp.arange(0, len(y))

    def log_likelihood(self, model_params, cv_fold):
        ll = st.norm.logpdf(self.y, loc=model_params["mu"], scale=1.0)
        return jnp.where(self.folds != cv_fold, ll, 0).sum()

    def log_prior(self, model_params):
        return st.norm.logpdf(model_params["mu"], loc=0.0, scale=1.0)

    def log_cond_pred(self, model_params, cv_fold):
        y_tilde = self.y[cv_fold]
        return st.norm.logpdf(y_tilde, loc=model_params["mu"], scale=1.0)

    def initial_value(self):
        return {"mu": 0.0}

    def cv_folds(self):
        return len(self.y)
```

Next, generate some data and fit the full-data model posterior.
```{python}
rng_key = random.PRNGKey(seed=42)
y = 2.5 + random.normal(rng_key, shape=(200,))
# create a model instance for this dataset
model = GaussianModel(y)
# fit full-data model
posterior = model.inference()
# display summary info
print(posterior)
```
```
Thor's Cross-Validatory Hammer
==============================

Starting Stan warmup using NUTS...
      500 warmup draws took 4.4 sec (114.3 iter/sec).
Running full-data inference with 8 chains...
      16,000 HMC draws took 2.0 sec (7,834 iter/sec).
Gaussian mean model inference summary
=====================================

16,000 draws from 2,000 iterations on 8 chains with seed 42

Parameter      Mean  (SE)      1%    5%    25%    Median    75%    95%    99%
-----------  ------  ------  ----  ----  -----  --------  -----  -----  -----
mu             2.61  (0.07)  2.45   2.5   2.57      2.61   2.66   2.73   2.78
```

Finally, run leave-one-out cross-validation:
```{python}
cv = posterior.cross_validate()
print(cv)
```
```
Cross-validation with 200 folds using 1,600 chains...
      3,200,000 HMC draws took 3.7 sec (876,199 iter/sec).

Cross-validation summary
========================

    elpd = -1.3272

Calculated from 200 folds (4 per fold, 800 total chains)

Average acceptance rate 76.4% (min 73.6%, max 80.0%)

Divergent chain count: 0
```
