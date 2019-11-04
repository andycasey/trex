
// Normal and logistic-truncated-log-normal mixture model

functions {
    real truncated_lognormal_lpdf(real y, real mu_multiple, real sigma_multiple, real mu_single) {
        real y_logit = 50 * (y - mu_single);
        return log_inv_logit(y_logit) + lognormal_lpdf(y | mu_multiple, sigma_multiple);
    }
}

data {
    int<lower=1> N; // number of data points
    real y[N]; // the data points.
    real bound_theta[2];
    real bound_mu_single[2];
    real bound_sigma_single[2];
    real bound_sigma_multiple[2];
    real mu_multiple_scalar;
}

parameters {
    real<lower=0.5, upper=1> theta; // the mixing parameter
    real<lower=bound_mu_single[1], upper=bound_mu_single[2]> mu_single; // single star distribution mean
    real<lower=bound_sigma_single[1], upper=bound_sigma_single[2]> sigma_single; // single star distribution sigma
    real<lower=bound_sigma_multiple[1], upper=bound_sigma_multiple[2]> sigma_multiple; // multiplcity log-normal distribution sigma
}

transformed parameters {
    real mu_multiple = log(mu_single + mu_multiple_scalar * sigma_single) + pow(sigma_multiple, 2);
}

model {
    for (n in 1:N) {
        target += log_mix(theta,
                          normal_lpdf(y[n] | mu_single, sigma_single),
                          truncated_lognormal_lpdf(y[n] | mu_multiple, sigma_multiple, mu_single));
    }
}