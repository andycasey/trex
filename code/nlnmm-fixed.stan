
// Normal and log-normal mixture model
// where the mu_multiple is fixed

data {
    int<lower=1> N; // number of data points
    real y[N]; // the data points.
    real scalar;
    real bound_theta[2];
    real bound_mu_single[2];
    real bound_sigma_single[2];
    real bound_sigma_multiple[2];
}

parameters {
    real<lower=fmax(0.0, bound_theta[1]), upper=fmin(1.0, bound_theta[2])> theta; // the mixing parameter
    real<lower=bound_mu_single[1], upper=bound_mu_single[2]> mu_single; // single star distribution mean
    real<lower=bound_sigma_single[1], upper=bound_sigma_single[2]> sigma_single; // single star distribution sigma
    real<lower=bound_sigma_multiple[1], upper=bound_sigma_multiple[2]> sigma_multiple; // multiplcity log-normal distribution sigma
}

transformed parameters {
    real mu_multiple;
    mu_multiple = log(mu_single + scalar * sigma_single) + pow(sigma_multiple, 2);
}

model {
    for (n in 1:N) {
        target += log_mix(theta,
                          normal_lpdf(y[n] | mu_single, sigma_single),
                          lognormal_lpdf(y[n] | mu_multiple, sigma_multiple));
    }
}