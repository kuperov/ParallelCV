rm(list=ls())

draws <- seq(1000, 40000, by=1000)

# log batch variance, using sqrt batch size rule
# http://personal.psu.edu/muh10/MCMCtut/batchmeans.R
log_bvar <- function(log_ypreds, ndraws) {
  # se of log ypred contribution
  vals <- rowSums(log_ypreds)
  b <- floor(sqrt(ndraws)) # batch size
  a <- floor(ndraws/b) # number of batches
  # logs of batch means
  logYs <- sapply(1:a,function(j) {
    jvals <- vals[((j-1)*b+1):(j*b)]
    maxval <- max(jvals)
    # log-mean-exp sum(log p(y))
    maxval + log(mean(exp(jvals-maxval)))
  })
  logMaxY <- max(logYs)
  logMuhat <- logMaxY + log(mean(exp(logYs - logMaxY)))
  # the following gives log(b*var(Y)), log of estimated sum of variances
  maxval <- max(logYs, logMuhat)
  log(b) + maxval + log(sum((exp(logYs - 0.5*maxval) - exp(logMuhat - 0.5*maxval))^2)) - log(a-1)
}

batch_var <- function(vals)
{
  N <- length(vals)
  b <- floor(sqrt(N)) # batch size
  a <- floor(N/b) # number of batches
  Ys <- sapply(1:a,function(k) return(mean(vals[((k-1)*b+1):(k*b)])))
  muhat <- mean(Ys)
  sigmahatsq <- b*sum((Ys-muhat)^2)/(a-1)
  sigmahatsq
}

elpdhats <- function(draw_file) {
  load(draw_file)
  K <- length(fold_draws)
  n <- 0
  for (k in seq_len(K)) {
    n <- n + ncol(fold_draws[[k]])
  }
  elpdhat_contribs <- matrix(NA, nrow=length(draws), ncol=n)
  logbvar <- rep(NA, length(draws))
  log_mcse <- rep(NA, length(draws))
  log_mcv <- rep(NA, length(draws))
  log_mcv_alt <- rep(NA, length(draws))
  log_mcse_alt <- rep(NA, length(draws))
  for (i in seq_along(draws)) {
    ndraws <- draws[i]
    offset <- 0
    logSigmahatsq <- rep(NA, K)  # fold log MC variances for density
    logSigmahatsq_alt <- rep(NA, K)  # fold MC variances of log density
    for (k in seq_len(K)) {
      this_n <- ncol(fold_draws[[k]])
      log_ypreds <- fold_draws[[k]][1:ndraws,]
      max_log_ypred <- apply(log_ypreds, 2, max)
      log_ypred <- max_log_ypred + log(mean(exp(log_ypreds - max_log_ypred)))
      elpdhat_contribs[i,offset+seq_len(this_n)] <- log_ypred
      offset <- offset + this_n
      logSigmahatsq[k] <- log_bvar(log_ypreds, ndraws)  # log of batch variance
      logSigmahatsq_alt[k] <- batch_var(log_ypred)  # batch variance of logs
    }
    # = log(b * var(exp(logYs)))
    maxlogShs <- max(logSigmahatsq)
    log_mcv[i] <- maxlogShs + log(sum(exp(logSigmahatsq - maxlogShs)))  # fold mc var
    log_mcse[i] <- 0.5 * (log_mcv[i] - log(ndraws))  # mcse for this fold
    log_mcv_alt[i] <- sum(logSigmahatsq_alt)  # alt. variance
    log_mcse_alt[i] <- sqrt(log_mcv_alt[i]/ndraws)  # alt. standard error
  }
  elpdhats <- rowSums(elpdhat_contribs)
  elpdhat_se <- sqrt(n) * apply(elpdhat_contribs, 1, sd)
  #mult <- max(elpdhats, log_mcse)
  # elpdhats_upper <- mult + log(exp(elpdhats - mult) + 1.96 * exp(log_mcse - mult))
  # elpdhats_lower <- mult + log(exp(elpdhats - mult) - 1.96 * exp(log_mcse - mult))
  elpdhats_upper <- log(exp(elpdhats) + 1.96 * exp(log_mcse))
  elpdhats_lower <- log(exp(elpdhats) - 1.96 * exp(log_mcse))
  # todo: use delta method to estimate cv error
  list(
    K=K,
    n=n,
    elpdhat=elpdhats,
    elpdhat_se=elpdhat_se,
    elpdhat_contribs=elpdhat_contribs,
    log_mcv=log_mcv,
    log_mcse=log_mcse,
    log_mcv_alt=log_mcv_alt,
    log_mcse_alt=log_mcse_alt,
    elpdhats_upper = elpdhats_upper,
    elpdhats_lower = elpdhats_lower
  )
}

compare <- function(model_A_file, model_B_file) {
  model_A <- elpdhats(model_A_file)
  model_B <- elpdhats(model_B_file)
  elpd_diff <- model_A$elpdhat - model_B$elpdhat
  elpd_diff_contribs <- model_A$elpdhat_contribs - model_B$elpdhat_contribs
  elpd_diff_se <- sqrt(nrow(elpd_diff_contribs)) * apply(elpd_diff_contribs, 1, sd)
  list(
    model_A=model_A,
    model_B=model_B,
    elpd_diff=elpd_diff,
    elpd_diff_se=elpd_diff_se
  )
}

cmp <- compare('R/fold_draws-A-long.Rda', 'R/fold_draws-B-long.Rda')
model_A <- cmp$model_A
model_B <- cmp$model_B
elpd_diff <- cmp$elpd_diff

par(mfrow=c(1,2))
drawsk <- draws*1e-3
plot(drawsk, model_A$elpdhat, type='l', lwd=2,
     main='Model elpdhats',
     xlab="draws ('000)", ylab='elpdhat',
     ylim=range(model_A$elpdhat+1.96*model_A$elpdhat_se,
                model_A$elpdhat-1.96*model_A$elpdhat_se,
                model_B$elpdhat+1.96*model_B$elpdhat_se,
                model_B$elpdhat-1.96*model_B$elpdhat_se
                ))
lines(drawsk, model_A$elpdhat+1.96*model_A$elpdhat_se, col='black', lty='dotted')
lines(drawsk, model_A$elpdhat-1.96*model_A$elpdhat_se, col='black', lty='dotted')
lines(drawsk, model_B$elpdhat, col='blue', lwd=2)
lines(drawsk, model_B$elpdhat+1.96*model_B$elpdhat_se, col='blue', lty='dotted')
lines(drawsk, model_B$elpdhat-1.96*model_B$elpdhat_se, col='blue', lty='dotted')
mtext("CV error bands: +/- 1.96 s.e.")
#legend('topright', c('Model A', 'Model B'), col=c('black','blue'), lwd=2, lty=1)

plot(drawsk, elpd_diff, type='l', col='red', lwd=2,
     main='Comparison (> 0 favors model A)',
     xlab="draws ('000)", ylab='elpd difference',
     ylim=range(elpd_diff+1.96*cmp$elpd_diff_se,elpd_diff-1.96*cmp$elpd_diff_se))
lines(drawsk, elpd_diff+1.96*cmp$elpd_diff_se, col='red', lty='dotted')
lines(drawsk, elpd_diff-1.96*cmp$elpd_diff_se, col='red', lty='dotted')
abline(h=0)
mtext("CV error bands: +/- 1.96 s.e.")
