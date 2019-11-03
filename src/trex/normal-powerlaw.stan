
// Mixture of a normal distribution and a power-law distribution

functions {
    // f(x, a) = a x^(a - 1)
    // log f(x, a) = log(a) + (a - 1) * log(x)

    real powerlaw_lpdf(real x, real a) {
        return log(a) + (a - 1) * log(x);
    }
}

data {
    int<lower=1> N; // number of data points
    real y[N]; // the data points.

    real bound_theta[2];
    real bound_mu_single[2];
    real bound_sigma_single[2];
    real bound_alpha[2];
}

transformed data {
    real delta = 1; // to prevent neg inf log probs at bounds
    real y_min = min(y) - delta;
    real y_max = max(y) + delta;
    real y_t[N];

    for (i in 1:N) {
        y_t[i] = (y[i] - y_min)/(y_max - y_min);
    }
}

parameters {
    real<lower=0, upper=1> theta; // the mixing parameter
    real<lower=0, upper=5> mu_single; // single star distribution mean
    real<lower=0.01, upper=3> sigma_single; // single star distribution sigma
    real<lower=3, upper=100> alpha;
}


model {
    for (n in 1:N) {
        target += log_mix(theta,
                          normal_lpdf(y[n] | mu_single, sigma_single),
                          beta_lpdf(y_t[n] | alpha, 1.0));
    }
}