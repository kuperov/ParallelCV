data {
  int<lower=0> p;
  int<lower=0> N;
  int<lower=0> Ntilde;
  matrix[N,p] X;
  vector[N] y;
  matrix[Ntilde,p] Xtilde;
  vector[Ntilde] ytilde;
}
parameters {
  real<lower=0> sigma;
  vector[p] beta;
}
model {
  beta ~ normal(1, 10);
  sigma ~ gamma(2, 2);
  y ~ normal(X * beta, sigma);
}
generated quantities {
  vector[Ntilde] log_ytilde;
  for (i in 1:Ntilde) {
    log_ytilde[i] = normal_lpdf(ytilde[i] | Xtilde[i] * beta, sigma);
  }
}
