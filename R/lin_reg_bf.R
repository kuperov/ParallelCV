library(cmdstanr)
library(stringr)

rm(list=ls())
set.seed(123)

N <- 100
beta0 <- c(1, 1, 1, 0.5)
p <- length(beta0)
sigma0 <- 1.
X <- matrix(rnorm(n=N*p), ncol = p, nrow = N)
y <- c(X %*% beta0 + rnorm(n=N, sd=sigma0))
mod <- cmdstan_model('R/lin_reg.stan')
K <- 5
rm(p)

inference_for_folds <- function(p, filename, chain_iter=10000) {
  fold_draws <- list()
  for (fold_id in 1:5) {
    indexes <- 1:N
    train_mask <- (indexes %% K) != (fold_id - 1)
    test_mask <- (indexes %% K) == (fold_id - 1)
    
    fit <- mod$sample(
      data = list(
        p=p,
        N=sum(train_mask),
        X=X[train_mask,1:p],
        y=y[train_mask],
        Ntilde=sum(test_mask),
        Xtilde=X[test_mask,1:p],
        ytilde=y[test_mask]
      ), 
      seed = 123,
      chains = 4, 
      parallel_chains = 4,
      refresh = 0,
      iter_sampling = chain_iter
    )
    
    fit$summary()
    
    draws <- fit$draws(format="draws_matrix")
    fold_draws[[fold_id]] <- draws[,str_starts(colnames(draws),'log_ytilde')]
  }
  save(fold_draws, file=filename)
}

inference_for_folds(p=4, chain_iter=10000, filename='R/fold_draws-A.Rda')
inference_for_folds(p=3, chain_iter=10000, filename='R/fold_draws-B.Rda')

inference_for_folds(p=4, chain_iter=100000, filename='R/fold_draws-A-long.Rda')
inference_for_folds(p=3, chain_iter=100000, filename='R/fold_draws-B-long.Rda')
