# script to generate test/gaussian_post.csv

import arviz as az
import numpy as np
import stan
from tabulate import tabulate

from ploo import GaussianModel

model = """// gaussian model for unit tests
data {
    int<lower=1> N;
    real Y[N];
}
parameters {
    real mu;
    real<lower=0> sigma;
}
model {
    mu ~ normal(0, 1);
    sigma ~ gamma(2, 2);  // params are shape and rate
    Y ~ normal(mu, sigma);
}
"""

# matches unit test test_log_lik
y = GaussianModel.generate(N=200, mu=0.5, sigma=2, seed=42)
dat1 = {"Y": np.array(y), "N": len(y)}
posterior = stan.build(model, data=dat1)
fit = posterior.sample(num_chains=4, num_samples=1000)
df = fit.to_frame()
df.to_csv("test/gaussian_post.csv")
post = az.from_pystan(fit)
az.summary(post)

rank = az.rhat(post, method="rank")
split = az.rhat(post, method="split")

rows = [[param, float(rank[param]), float(split[param])] for param in ["mu", "sigma"]]
tbl = tabulate(rows, headers=["Parameter", "r.n. Rhat", "split Rhat"])
print(tbl)
