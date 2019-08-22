

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LogNorm

import pystan as stan



np.random.seed(1)

observing_span = 668 # u.days


N_simulations = 10000

radial_velocity = lambda t, P, K, varphi=0: K * np.sin(2 * np.pi * t / P + varphi)


P = 10**np.random.uniform(-1.5, np.log10(0.1 * observing_span))
K = 10**np.random.uniform(-0.5, 3)

varphi = np.random.uniform(0, 2 * np.pi)

N = np.random.randint(3, 30)

t = np.random.uniform(0, observing_span, N)

v_systemic = np.random.normal(0, 100)

v = v_systemic + radial_velocity(t, P, K, varphi)

model_code = """
data {
    int<lower=2> N;
    real t[N];
    real radial_velocity;
    real radial_velocity_jitter;
}
parameters {
    real v_systemic;
    real<lower=0> K;
    real<lower=0,upper=668> P;
    real<lower=0, upper=2*3.14> varphi;
}

transformed parameters {

    real est_mu;
    real<lower=0> est_sigma;
    {

        real v[N];
        for (i in 1:N)
            v[i] = v_systemic + K * sin(2 * 3.14 * t[i]/P + varphi);

        est_mu = sum(v)/N;

        est_sigma = 0;
        for (i in 1:N)
            est_sigma = est_sigma + (v[i] - est_mu) * (v[i] - est_mu);

        est_sigma = sqrt(est_sigma/(N - 1));
    }

}

model {
    radial_velocity ~ normal(est_mu, 0.01);
    radial_velocity_jitter ~ normal(est_sigma, 0.01);
}
"""

model = stan.StanModel(model_code=model_code)

data = dict(N=N, t=t, radial_velocity=np.mean(v), radial_velocity_jitter=np.std(v))
p_opt = model.optimizing(data=data)

chains = 2
samples = model.sampling(init=[p_opt] * chains, chains=chains, data=data, iter=10000)

fig = samples.plot()
fig.tight_layout()


chains = 2
truth = dict(v_systemic=v_systemic, K=K, P=P, vaphi=varphi)
samples = model.sampling(init=[truth] * chains, chains=chains, data=data, iter=10000)

fig = samples.plot()
fig.tight_layout()


raise a




data = dict(N=N, t=t, mu=v_systemic, rv_jitter=np.std(v))
p_opt = model.optimizing(data=data)


chains = 2
truth = dict(K=K, P=P, vaphi=varphi, model_rv_jitter=np.std(v))
samples = model.sampling(init=[truth]*chains, chains=chains, data=data, iter=100000)

fig = samples.plot()
fig.tight_layout()


raise a



model_code = """
data {
    int<lower=2> N;
    real t[N];
    real rv_jitter;
}
parameters {
    real<lower=0> K;
    real<lower=0,upper=668> P;
    real<lower=0, upper=2*3.14> varphi;
}

transformed parameters {

    real model_rv_jitter;
    {
        real v[N];
        real mean_v;

        for (i in 1:N)
            v[i] = K * sin(2 * 3.14 * t[i]/P + varphi);
        
        mean_v = sum(v)/N;
        model_rv_jitter = 0;
        for (i in 1:N)
            model_rv_jitter = model_rv_jitter + (v[i] - mean_v) * (v[i] - mean_v);

        model_rv_jitter = sqrt(model_rv_jitter/(N - 1));
    }
}

model {
    rv_jitter ~ normal(model_rv_jitter, 0.1);
}
"""




model = stan.StanModel(model_code=model_code)



data = dict(N=N, rv_jitter=np.std(v), t=t)
p_opt = model.optimizing(data=data)


chains = 2
truth = dict(K=K, P=P, vaphi=varphi, model_rv_jitter=np.std(v))
samples = model.sampling(init=[truth]*chains, chains=chains, data=data, iter=100000)

fig = samples.plot()
fig.tight_layout()

