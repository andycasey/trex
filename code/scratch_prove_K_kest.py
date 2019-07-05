
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LogNorm

np.random.seed(0)

observing_span = 668 # u.days

"""
N_simulations = 10000

P_true = 10**np.random.uniform(-1.5, 6, size=N_simulations) # u.days
P_true = 10**np.random.uniform(-1.5, np.log10(0.1 * observing_span), size=N_simulations)

K_true = 10**np.random.uniform(-0.5, 3, size=N_simulations) # u.km/u.s
"""

from astropy.table import Table
sb9 = Table.read("../data/catalogs/sb9-pourbaix-et-al-orbits.fits")
P_true = np.array(sb9["Per"])
K_true = np.array(sb9["K1"])

keep = np.isfinite(K_true * P_true) \
     * (P_true < (2 * observing_span))
P_true, K_true = (P_true[keep], K_true[keep])

N_simulations = len(P_true)

varphi_true = np.random.uniform(0, 2 * np.pi, size=N_simulations)


Ns = np.random.randint(5, 15, size=N_simulations)
times = np.random.uniform(0, observing_span, size=(N_simulations, max(Ns)))

v_sys_true = np.random.normal(0, 100, size=N_simulations)
#v_sys_true = np.zeros(N_simulations)

radial_velocity = lambda t, P, K, varphi=0: K * np.sin(2 * np.pi * t / P + varphi)

K_est = np.zeros_like(K_true)
K_est_err = np.zeros_like(K_true)

V_rms = np.zeros_like(K_true)
V_std = np.zeros_like(K_true)

#for i, (P, K) in enumerate(tqdm(zip(P_true, K_true), total=N_simulations)):
for i, (P, K, varphi, N, t_) \
in enumerate(tqdm(zip(P_true, K_true, varphi_true, Ns, times), total=N_simulations)):
    
    t = t_[:N]
    v = v_sys_true[i] + radial_velocity(t, P, K, varphi)
    v += np.random.normal(0, 0.1, size=v.size)

    V_rms[i] = np.sqrt(np.sum(v**2)/N)
    V_std[i] = np.sqrt(np.sum((v - np.mean(v))**2)/(N-1))


    K_est[i] = np.sqrt(2) * V_std[i]

    K_est_err[i] = np.sqrt(K_est[i])

    """
    K_trials = np.sqrt(np.var(v) * (N - 1) \
             / np.sum(np.sin(np.random.uniform(0, 2 * np.pi, size=(N, 1000)))**2, axis=0))

    K_est_new[i] = (np.mean(K_trials), np.std(K_trials))
    """


fig, ax = plt.subplots()
ax.scatter(K_true, P_true, s=1)
ax.loglog()

ax.set_xlim(np.min(K_true), np.max(K_true))
ax.set_ylim(np.min(P_true), np.max(P_true))


fig, ax = plt.subplots()
scat = ax.scatter(K_true, K_est, c=P_true, norm=LogNorm(), s=10)
ax.errorbar(K_true, K_est, yerr=K_est_err, fmt="none", ecolor="#666666", zorder=-1, lw=0.5)
ax.set_xlabel(r"$K_\mathrm{true}$ / km\,s$^{-1}$")
ax.set_ylabel(r"$K_\mathrm{est}$ / km\,s$^{-1}$")
ax.loglog()

limits = np.vstack([K_true, K_est]).flatten()
limits = (np.min(limits), np.max(limits))

ax.set_xlim(limits)
ax.set_ylim(limits)    

cbar = plt.colorbar(scat)


fig, ax = plt.subplots()
scat = ax.scatter(K_true, K_est, c=Ns, s=10)
ax.errorbar(K_true, K_est, yerr=K_est_err, fmt="none", ecolor="#666666", zorder=-1, lw=0.5)
ax.set_xlabel(r"$K_\mathrm{true}$ / km\,s$^{-1}$")
ax.set_ylabel(r"$K_\mathrm{est}$ / km\,s$^{-1}$")
ax.loglog()

limits = np.vstack([K_true, K_est]).flatten()
limits = (np.min(limits), np.max(limits))

ax.set_xlim(limits)
ax.set_ylim(limits)    

cbar = plt.colorbar(scat)

diff = K_true - K_est
print(f"Estimator mean: {np.mean(diff):.2f}, median: {np.median(diff):.2f}, std. dev.: {np.std(diff):.2f}")



fig, ax = plt.subplots()
ax.scatter(K_true, diff, s=10)

Q = (K_true - K_est)/K_est_err

fig, ax = plt.subplots()
ax.hist(Q, bins=np.linspace(-5, 5, 25), normed=True)
xi = np.linspace(-5, 5, 1000)
from scipy import stats
yi = stats.norm.pdf(xi, 0, 1)

ax.plot(xi, yi)


raise a



K_est_err = 0.09 * np.abs(K_est)
Q = diff/K_est_err

fig, ax = plt.subplots()
ax.hist(Q, bins=np.linspace(-5, 5, 25), normed=True)
xi = np.linspace(-5, 5, 1000)
from scipy import stats
yi = stats.norm.pdf(xi, 0, 1)

ax.plot(xi, yi)

raise a


fig, ax = plt.subplots()
ax.scatter(K_true, diff/K_true, s=1)

raise a



fig, axes = plt.subplots(1, 2)

scat = axes[0].scatter(K_true, K_est - K_true, s=1, c=P_true, norm=LogNorm())
axes[1].scatter(K_true, K_est_new.T[0] - K_true, s=1, c=P_true, norm=LogNorm())
axes[1].errorbar(K_true, K_est_new.T[0] - K_true, yerr=K_est_new.T[1], fmt="none", ecolor="#666666", zorder=-1, lw=0.5)

ax.set_xlabel(r"$K_\mathrm{true}$ / km\,s$^{-1}$")
ax.set_ylabel(r"$\Delta{}K$ / km\,s$^{-1}$")

print(f"Old estimator mean: {np.mean(K_true - K_est):.2f} and median: {np.median(K_true - K_est):.2f} and stddev: {np.std(K_true - K_est):.2f}")
print(f"New estimator mean: {np.mean(K_true - K_est_new.T[0]):.2f} and median: {np.median(K_true - K_est_new.T[0]):.2f} and stddev: {np.std(K_true - K_est_new.T[0]):.2f}")

Q = (K_est_new.T[0] - K_true)/K_est_new.T[1]

fig, ax = plt.subplots()
ax.hist(Q, bins=np.linspace(-3, 3, 25))

raise a
#ax.semilogx()


cbar = plt.colorbar(scat)
