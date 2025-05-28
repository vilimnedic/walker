//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N; 
  int<lower=0> K;
  int<lower=0> y[N];
  matrix[N, K] X;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters { 
  vector[K] beta; 
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  beta ~ normal(0, 5);
  target += poisson_log_glm_lpmf(y | X[, 2:K], beta[1], beta[2:K]);
}

// poisson_glm.stan

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    real eta_n = dot_product(row(X, n), beta);
    log_lik[n] = poisson_log_lpmf(y[n] | eta_n);
    y_rep[n]   = poisson_log_rng(eta_n);
  }
}