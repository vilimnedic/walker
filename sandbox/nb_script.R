library(rstan)
library(bayesplot)
library(loo)
library(posterior)
library(MASS)          

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# DGP
set.seed(123)
N  <- 100
K  <- 3 # predicors incl intercept
X  <- cbind(1, matrix(rnorm(N*(K-1)), N, K-1))
beta_true <- c(-0.2, 0.5, -1.1)
phi_true  <- 2  # NB Size
eta <- X %*% beta_true
mu  <- exp(eta)               
y   <- rnbinom(N, size = phi_true, mu = mu)

dat <- list(N = N, K = K, y = y, X = X)

# Stan Man NB
fit_man <- stan("sandbox/nb2_man.stan",
                data   = dat,
                chains = 4, iter = 2000, seed = 123, refresh = 500, model_name = "NB2_man")
print(fit_man, pars = c("beta","phi"))

# Stan GLM
fit_glm <- stan("sandbox/nb2_glm.stan",
                data   = dat,
                chains = 4, iter = 2000, seed = 123, refresh = 500, model_name = "NB2_glm")
print(fit_glm, pars = c("beta","phi"))


# Stan Poisson
fit_poi <- stan("sandbox/nb2_poi.stan",
                data   = dat,
                chains = 4, iter = 2000, seed = 123, refresh = 500, model_name = "NB2_poi")
print(fit_poi, pars = c("beta","phi"))

summary(fit_poi, pars = "lambda")$summary[, c("n_eff", "Rhat")]
traceplot(fit_poi, pars = c("beta", "phi"))
mcmc_trace(as_draws_array(fit_poi), regex_pars = "beta\\[|phi")

# Built-in R
fit_glm_r <- glm.nb(y ~ X[,-1])
summary(fit_glm_r)

# Posterior checks
posterior::summarise_draws(as_draws_df(fit_man, pars = c("beta","phi")))
posterior::summarise_draws(as_draws_df(fit_glm, pars = c("beta","phi")))
posterior::summarise_draws(as_draws_df(fit_poi, pars = c("beta","phi")))

yrep_man <- rstan::extract(fit_man, "y_rep")$y_rep
yrep_glm <- rstan::extract(fit_glm, "y_rep")$y_rep
yrep_poi <- rstan::extract(fit_poi, "y_rep")$y_rep

ppc_dens_overlay(y, yrep_man[1:200,])
ppc_dens_overlay(y, yrep_glm[1:200,])
ppc_dens_overlay(y, yrep_poi[1:200,])

log_lik_man <- extract_log_lik(fit_man)
log_lik_glm <- extract_log_lik(fit_glm)
log_lik_poi <- extract_log_lik(fit_poi)
loo_compare(loo(log_lik_man), loo(log_lik_glm), loo(log_lik_poi))
