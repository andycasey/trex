
// Normal and log-normal mixture model, where the log normal is truncated on the left with a sigmoid

functions {
    
    real truncated_lognormal_lpdf(real y, real y_logit, real mu_multiple, real sigma_multiple) {
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
    real<lower=(0.125 * mu_single), upper=bound_sigma_single[2]> sigma_single; // single star distribution sigma
    real<lower=(sigma_single/(mu_single + sigma_single)), upper=bound_sigma_multiple[2]> sigma_multiple; // multiplcity log-normal distribution sigma
}

transformed parameters {
    real mu_multiple = log(mu_single + mu_multiple_scalar * sigma_single) + pow(sigma_multiple, 2);
    real y_logit[N];
    {
        int M = 2;
        real sigmoid_weight = (1/sigma_single) * log(pow(2 * pi() * sigma_single, 0.5) * exp(0.5 * pow(M, 2)) - 1);

        for (n in 1:N) {
            y_logit[n] = sigmoid_weight * (y[n] - mu_single);
        }
    
    }
    
}

model {
    for (n in 1:N) {
        target += log_mix(theta,
                          normal_lpdf(y[n] | mu_single, sigma_single),
                          truncated_lognormal_lpdf(y[n] | y_logit[n], mu_multiple, sigma_multiple));
    }
}

generated quantities {
    real ll_s[N];
    real ll_m[N];

    for (n in 1:N) {
        ll_s[n] = normal_lpdf(y[n] | mu_single, sigma_single);
        ll_m[n] = truncated_lognormal_lpdf(y[n] | y_logit[n], mu_multiple, sigma_multiple);
    }
}