
"""
What's better: \sigma_{MTA} or \sigma_{RV}?
"""


import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G
from scipy import stats
from tqdm import tqdm

import stan_utils as stan


np.random.seed(0)

N = 128
v_intrinsic_err = 0.5 # km/s

log_P_min, log_P_max = (-2, 12) # log_10(P / day)
q_min, q_max = (0.1, 1)
M_min, M_max = (0.1, 100) # sol masses

observing_span_in_days = 668

data_kwds = dict(scalar=1, 
                 bound_theta=(0.25, 1),
                 bound_mu_single=(0.1, 15),
                 bound_sigma_single=(0.1, 5),
                 bound_sigma_multiple=(0.2, 1))

init_kwds = dict(theta=0.75, sigma_single=0.3, sigma_multiple=0.5)


def draw_periods(N, log_P_mu=4.8, log_P_sigma=2.3, log_P_min=-2, log_P_max=12):

    # orbital period Duquennoy and Mayor 1991 distribution
    P = np.empty(N)
    P_min, P_max = (10**log_P_min, 10**log_P_max)

    for i in range(N):
        while True:
            x = np.random.lognormal(log_P_mu, log_P_sigma)
            if P_max >= x >= P_min:
                P[i] = x
                break
    return P


def mass_ratios(M_1, q_min=0.1, q_max=1):
    N = len(M_1)
    q = np.empty(N)
    for i, M in enumerate(M_1):
        q[i] = np.random.uniform(q_min / M.value, q_max)
    return q

def salpeter_imf(N, alpha=2.35, M_min=0.1, M_max=100):
    # salpeter imf
    log_M_limits = np.log([M_min, M_max])

    max_ll = M_min**(1.0 - alpha)

    M = []
    while len(M) < N:
        Mi = np.exp(np.random.uniform(*log_M_limits))

        ln = Mi**(1 - alpha)
        if np.random.uniform(0, max_ll) < ln:
            M.append(Mi)

    return np.array(M)





# Draw periods, masses, and q values.        
P = draw_periods(N, log_P_min=log_P_min, log_P_max=log_P_max) * u.day
M_1 = salpeter_imf(N, M_min=M_min, M_max=M_max) * u.solMass
q = mass_ratios(M_1, q_min=q_min, q_max=q_max)
M_2 = q * M_1
e = np.zeros(N)
i = np.arccos(np.random.uniform(0, 1, size=N))

# Compute the semi-major axis given the orbital period and total mass.
M_total = M_1 + M_2
a = np.cbrt(G * (M_1 + M_2) * (P/(2 * np.pi))**2)

# Compute the radial velocity semi-amplitude.
K = 2 * np.pi * a * np.sin(i) / (P * np.sqrt(1 - e**2))

# Get K and P together.
K = K.to(u.km/u.s).value
P = P.to(u.day).value

# Get the number of transits.
sources = h5.File("../data/sources.hdf5", "r")
rv_nb_transits = sources["sources/rv_nb_transits"][()]
rv_nb_transits = rv_nb_transits[rv_nb_transits > 0]
rv_nb_transits = np.random.choice(rv_nb_transits, N, replace=False)

# Let's say that all stars have exactly the same colour, apparent magnitude, and absolute magnitude.


sigma_mta_to_sigma_rv = lambda v_std, O: np.sqrt(((2 * O)/np.pi) * v_std**2 - 0.11**2)

# Do single stars first.
single_sigma_rv = np.empty((N))
single_sigma_mta = np.empty((N))

for i, O in tqdm(enumerate(rv_nb_transits)):

    v = np.random.normal(0, v_intrinsic_err, size=O)
    v_std = np.std(v)

    single_sigma_mta[i] = v_std
    single_sigma_rv[i] = sigma_mta_to_sigma_rv(v_std, O)

    assert np.isfinite(v_std)


# Do binary stars.
binary_sigma_rv = np.empty((N))
binary_sigma_mta = np.empty((N))

for i, O in tqdm(enumerate(rv_nb_transits)):

    t = np.random.uniform(0, observing_span_in_days, size=O)
    phi = np.random.uniform(0, 2 * np.pi)

    v_t = K[i] * np.sin(2 * np.pi * t / P[i] + phi)
    v = np.random.normal(v_t, v_intrinsic_err)

    v_std = np.std(v)

    binary_sigma_mta[i] = v_std
    binary_sigma_rv[i] = sigma_mta_to_sigma_rv(v_std, O)

    assert np.isfinite(v_std)


print(f"Number of single stars with finite values: {np.isfinite(single_sigma_rv).sum()}")
print(f"Number of binary stars with finite values: {np.isfinite(binary_sigma_rv).sum()}")

# Now let us imagine that we were fitting these data with a mixture model in order to estimate the
# intrinsic uncertainty.
model = stan.load_stan_model("nlnmm-fixed.stan")

y_rv = np.hstack([single_sigma_rv, binary_sigma_rv])
y_rv = y_rv[np.isfinite(y_rv)]

data_dict = dict(y=y_rv, N=y_rv.size)
data_dict.update(data_kwds)

init_dict = dict(mu_single=np.min([np.median(y_rv, axis=0), 10]))
init_dict.update(init_kwds)

p_opt_rv = model.optimizing(data=data_dict, init=init_dict)

fig, ax = plt.subplots()
ax.hist(y_rv, bins=np.linspace(0, 10, 100))



y_mta = np.hstack([single_sigma_mta, binary_sigma_mta])
y_mta = y_mta[np.isfinite(y_mta)]
data_dict = dict(y=y_mta, N=y_mta.size)
data_dict.update(data_kwds)

init_dict = dict(mu_single=np.min([np.median(y_mta, axis=0), 10]))
init_dict.update(init_kwds)

p_opt_mta = model.optimizing(data=data_dict, init=init_dict)

fig, ax = plt.subplots()
ax.hist(y_mta, bins=np.linspace(0, 10, 100))



mta_color, rv_color = ("#A62E2E", "#F2884B")
fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.75))
ax = axes[0]
ax.hist([y_rv, y_mta], color=[rv_color, mta_color],
        bins=np.linspace(0, 100, 50), histtype="stepfilled", alpha=0.5)


for ax in axes:
    ax.axvline(v_intrinsic_err, c="#000000", lw=2, zorder=100)
    ax.axvline(init_dict["mu_single"], c="#000000", lw=0.5, linestyle=":", zorder=100)


ax = axes[1]

xi = np.linspace(0, max(5 * v_intrinsic_err, 10), 1000)
p_rv = stats.norm.pdf(xi, p_opt_rv["mu_single"], p_opt_rv["sigma_single"])
p_mta = stats.norm.pdf(xi, p_opt_mta["mu_single"], p_opt_mta["sigma_single"])

ax.fill_between(xi, 0, p_rv, facecolor=rv_color, alpha=0.5, label="rv")
ax.fill_between(xi, 0, p_mta, facecolor=mta_color, alpha=0.5, label="mta")
ax.plot(xi, p_rv, c=rv_color, lw=2, zorder=10)
ax.plot(xi, p_mta, c=mta_color, lw=2, zorder=10)

ax.legend(frameon=False)
ax.set_ylim(0, ax.get_ylim()[1])

plt.show()