
data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=0> y[N];
  matrix[N, K] X;
}


parameters {
  vector[K] beta; 
  real<lower=0> phi;
  vector<lower=0>[N] eps;
  // vector<lower=0>[N] lambda;
}

model {
  beta ~ normal(0, 5);
  phi ~ exponential(1);
  vector[N] mu = exp(X * beta);

  for (n in 1:N) {
    eps[n] ~ gamma(phi,  phi);
    y[n] ~ poisson(mu[n] * eps[n]);
  }
}

