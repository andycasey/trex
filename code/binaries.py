

import numpy as np


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