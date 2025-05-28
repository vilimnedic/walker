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
  int<lower=1>  N;
  int<lower=1>  K;
  int<lower=0> y[N];
  matrix[N, K] X;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[K] beta;
  real<lower=0> phi;
}

transformed parameters {
  vector[N] mu = exp(X * beta);
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  beta ~ normal(0, 5);
  phi ~ exponential(1);

  target += neg_binomial_2_log_glm_lpmf(y | X[,2:K], beta[1], beta[2:K], phi);
}

generated quantities {
  int<lower=0> y_rep[N];
  vector[N] log_lik;

  for (n in 1:N) {
    y_rep[n] = neg_binomial_2_rng(mu[n], phi);     
    log_lik[n] = neg_binomial_2_lpmf(y[n] | mu[n], phi);
  }
}
