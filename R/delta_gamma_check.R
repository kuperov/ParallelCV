# check delta method for variance
set.seed(123)

J <- 10
measured_var <- rep(NA, J)
an_log_var <- rep(NA, J)
delta_var <- rep(NA, J)
ns <- 2^(seq_len(J)+4)
for (j in seq_len(J)) {
  reps <- 1000
  n <- ns[j]
  means <- rep(NA, reps)
  for (i in seq_len(reps)) {
    xs <- rgamma(n=n, shape=2, rate=1)
    means[i] <- mean(xs)
  }
  # analytical variance
  gamma_var <- 2/1^2
  gamma_mean <- 2/1
  an_log_var[j] <- gamma_var/(gamma_mean^2)/n
  # empirical variance
  log_means <- log(means)
  measured_var[j] <- var(log_means)
  # delta plug-in estimate
  delta_var[j] <- var(xs)/(mean(xs)^2)/n
}
par(mfrow=c(1,1))
plot(ns, an_log_var, col='red', type='l',
     log='xy', main='Delta method approximation check',
     ylab='Variance (log scale)', xlab='n')
lines(ns, measured_var, col='darkorange')
lines(ns, delta_var, col='blue')
legend('topright', c('analytical','empirical','delta'), col=c('red','darkorange','blue'), lty=1)
