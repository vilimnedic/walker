library(rstan)
library(bayesplot)
library(loo)
library(posterior)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# DGP 
set.seed(123)
N <- 100
K <- 3 # predicors incl intercept
X <- cbind(1, matrix(rnorm(N*(K-1)), N, K-1))
beta_true <- c(-0.2, 0.5, -1.1)
eta <- X %*% beta_true 
lambda <- exp(eta)
y <- rpois(N, lambda)

dat <- list(N = N, K = K, y = y, X = X)

# Stan man
fit_man <- stan("sandbox/poisson_man.stan", data = dat,
                   chains = 4, iter = 2000, seed = 123, verbose = TRUE)
print(fit_man, pars = c("beta"))

# Stan GLM
fit_glm <- stan("sandbox/poisson_glm.stan", data = dat,
                chains = 4, iter = 2000, seed = 123, verbose = TRUE)
print(fit_glm, pars = c("beta"))

# BUilt-in R
fit_glm_r <- glm(y ~ X[, -1], family = poisson(link = "log"))
summary(fit_glm_r)


# Posterior checks
posterior::summarise_draws(as_draws_df(fit_man, pars = "beta"))
posterior::summarise_draws(as_draws_df(fit_glm, pars = "beta"))

yrep_man <- extract(fit_man, "y_rep")$y_rep
yrep_glm <- extract(fit_glm, "y_rep")$y_rep
ppc_dens_overlay(y, yrep_man[1:200, ])    
ppc_dens_overlay(y, yrep_glm[1:200, ])
ppc_stat(y, yrep_man, stat = "mean")
ppc_stat(y, yrep_glm, stat = "mean")           

log_lik_man <- extract_log_lik(fit_man)
log_lik_glm <- extract_log_lik(fit_glm)
loo(log_lik_man)
loo_compare(loo(log_lik_man), loo(log_lik_glm)) 
