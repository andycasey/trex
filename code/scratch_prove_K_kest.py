
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LogNorm

np.random.seed(0)

observing_span = 668 # u.days


N_simulations = 10000



P_true = 10**np.random.uniform(-1.5, 6, size=N_simulations) # u.days
P_true = 10**np.random.uniform(-1.5, np.log10(0.1 * observing_span), size=N_simulations)

K_true = 10**np.random.uniform(-0.5, 3, size=N_simulations) # u.km/u.s
varphi_true = np.random.uniform(0, 2 * np.pi, size=N_simulations)



Ns = np.random.randint(10, 100, size=N_simulations)
times = np.random.uniform(0, observing_span, size=(N_simulations, max(Ns)))

radial_velocity = lambda t, P, K, varphi=0: K * np.sin(2 * np.pi * t / P + varphi)

K_est = np.zeros_like(K_true)
K_est_new = np.zeros((N_simulations, 2))

#for i, (P, K) in enumerate(tqdm(zip(P_true, K_true), total=N_simulations)):
for i, (P, K, varphi, N, t_) \
in enumerate(tqdm(zip(P_true, K_true, varphi_true, Ns, times), total=N_simulations)):
    
    t = t_[:N]
    v = radial_velocity(t, P, K, varphi)

    # TODO: noiseless obs?

    K_est[i] = np.sqrt(2) * np.std(v)


    K_trials = np.sqrt(np.var(v) * (N - 1) \
             / np.sum(np.sin(np.random.uniform(0, 2 * np.pi, size=(N, 1000)))**2, axis=0))

    K_est_new[i] = (np.mean(K_trials), np.std(K_trials))


    if i == 0:

        ti = np.linspace(0, observing_span, 1000)
        vi = radial_velocity(ti, P, K, varphi)

        fig, ax = plt.subplots()
        ax.scatter(t, v)
        ax.plot(ti, vi, c="#000000", zorder=-1, linewidth=0.5)

        fig, ax = plt.subplots()
        _, bins, __ = ax.hist(vi, bins=25, normed=True, alpha=0.5, facecolor="tab:blue")
        ax.hist(v, bins=bins, normed=True, alpha=0.5, facecolor="tab:red")





fig, ax = plt.subplots()
ax.scatter(K_true, P_true)
ax.loglog()

ax.set_xlim(np.min(K_true), np.max(K_true))
ax.set_ylim(np.min(P_true), np.max(P_true))


fig, ax = plt.subplots()
scat = ax.scatter(K_true, K_est, c=P_true, norm=LogNorm())
ax.set_xlabel(r"$K_\mathrm{true}$ / km\,s$^{-1}$")
ax.set_ylabel(r"$K_\mathrm{est}$ / km\,s$^{-1}$")
ax.loglog()

limits = np.vstack([K_true, K_est]).flatten()
limits = (np.min(limits), np.max(limits))

ax.set_xlim(limits)
ax.set_ylim(limits)    

cbar = plt.colorbar(scat)



fig, axes = plt.subplots(1, 2)

scat = axes[0].scatter(K_true, K_est - K_true, s=1, c=P_true, norm=LogNorm())
axes[1].scatter(K_true, K_est_new.T[0] - K_true, s=1, c=P_true, norm=LogNorm())
axes[1].errorbar(K_true, K_est_new.T[0] - K_true, yerr=K_est_new.T[1], fmt="none", ecolor="#666666", zorder=-1)

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
