
import warnings
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from json import dumps

def repr_dict(d):
    return dumps(d, indent=2, sort_keys=True)


def normal_lpdf(y, mu, sigma):
    ivar = sigma**(-2)
    return 0.5 * (np.log(ivar) - np.log(2 * np.pi) - (y - mu)**2 * ivar)

def lognormal_lpdf(y, mu, sigma):
    ivar = sigma**(-2)
    return - 0.5 * np.log(2 * np.pi) - np.log(y * sigma) \
           - 0.5 * (np.log(y) - mu)**2 * ivar



# Calculate log-probabilities for all of the stars we considered.
def membership_probability(y, p_opt):

    y = np.atleast_1d(y)
    theta, s_mu, s_sigma, m_mu, m_sigma = _unpack_params(_pack_params(**p_opt))

    assert s_mu.size == y.size, "The size of y should match the size of mu"


    D = y.size
    ln_prob = np.zeros((D, 2))
    for d in range(D):
        ln_prob[d] = [
            normal_lpdf(y[d], s_mu[d], s_sigma[d]),
            lognormal_lpdf(y[d], m_mu[d], m_sigma[d])
        ]

    # TODO: I am not certain that I am summing these log probabilities correctly

    sum_ln_prob = np.sum(ln_prob, axis=0) # per mixture
    ln_likelihood = logsumexp(sum_ln_prob)

    with np.errstate(under="ignore"):
        ln_membership = sum_ln_prob - ln_likelihood

    return np.exp(ln_membership)


# Stan needs a finite value to initialize correctly, so we will use a dumb (more
# robust) optimizer to get an initialization value.
def norm_pdf(x, norm_mu, norm_sigma, theta):
    return theta * (2 * np.pi * norm_sigma**2)**(-0.5) * np.exp(-(x - norm_mu)**2/(2*norm_sigma**2))
    

def lognorm_pdf(x, lognorm_mu, lognorm_sigma, theta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pdf = (1.0 - theta)/(x * lognorm_sigma * np.sqrt(2*np.pi)) \
               * np.exp(-0.5 * ((np.log(x) - lognorm_mu)/lognorm_sigma)**2)

    return pdf


def f(y, w, s_mu, s_sigma, b_mu, b_sigma):

    y = np.atleast_1d(y)

    s_ivar, b_ivar = (s_sigma**-2, b_sigma**-2)

    hl2p = 0.5 * np.log(2*np.pi)

    s_lpdf = np.log(w) \
           - 0.5 * (y - s_mu)**2 * s_ivar + 0.5 * np.log(s_ivar) - hl2p
    b_lpdf = np.log(1 - w) \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar - np.log(y * b_sigma) - hl2p

    ll = np.sum(logsumexp([s_lpdf, b_lpdf], axis=0))

    return ll


def g(y, w, s_mu, s_sigma, b_mu, b_sigma):

    df_dw = np.log(np.sum(np.exp(1/w) - np.exp(1/(1-w))))
    return df_dw




def ln_likelihood(y, theta, s_mu, s_sigma, b_mu, b_sigma):
    
    s_ivar = s_sigma**-2
    b_ivar = b_sigma**-2
    hl2p = 0.5 * np.log(2*np.pi)
    
    s_lpdf = -hl2p + 0.5 * np.log(s_ivar) \
           - 0.5 * (y - s_mu)**2 * s_ivar
    
    b_lpdf = -np.log(y*b_sigma) - hl2p \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar

    foo = np.vstack([s_lpdf, b_lpdf]).T + np.log([theta, 1-theta])
    ll = np.sum(logsumexp(foo, axis=1))

    #ll = np.sum(s_lpdf) + np.sum(b_lpdf)

    ##print(lpdf)
    
    #assert np.isfinite(ll)
    return ll


def ln_prior(theta, s_mu, s_sigma, b_mu, b_sigma, bounds=None):

    # Ensure that the *mode* of the log-normal distribution is larger than the
    # mean of the normal distribution
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_mu_multiple = np.log(s_mu + s_sigma) + b_sigma**2


    if not (1 >= theta >= 0) \
    or not (15 >= s_mu >= 0.5) \
    or not (10 >= s_sigma >= 0.05) \
    or not (1.6 >= b_sigma >= 0.20) \
    or np.any(b_mu < min_mu_multiple):
        return -np.inf
    """
    if bounds is not None:
        vals = dict(theta=theta, mu_single=s_mu, sigma_single=s_sigma, mu_multiple=b_mu, sigma_multiple=b_sigma)
        for k, (lower, upper) in bounds.items():
            if k.startswith("bound_"):
                k = "_".join(k.split("_")[1:])
            if not ((upper >= vals[k]) and (vals[k] >= lower)):
                return -np.inf


    # Beta prior on theta.
    return 0 #stats.beta.logpdf(theta, 5, 5)


def ln_prob(y, L, *params, bounds=None):
    theta, s_mu, s_sigma, b_mu, b_sigma = _unpack_params(params, L=L)
    lp = ln_prior(theta, s_mu, s_sigma, b_mu, b_sigma, bounds=bounds)
    if np.isfinite(lp):
        return lp + ln_likelihood(y, theta, s_mu, s_sigma, b_mu, b_sigma)
    return lp


def _unpack_params(params, L=None):
    # unpack the multdimensional values.
    if L is None:
        L = int((len(params) - 1)/4)

    theta = params[0]
    mu_single = np.array(params[1:1 + L])
    sigma_single = np.array(params[1 + L:1 + 2 * L])
    mu_multiple = np.array(params[1 + 2 * L:1 + 3 * L])
    sigma_multiple = np.array(params[1 + 3 * L:1 + 4 * L])

    return (theta, mu_single, sigma_single, mu_multiple, sigma_multiple)


def _pack_params(theta, mu_single, sigma_single, mu_multiple, sigma_multiple, mu_multiple_uv=None, **kwargs):
    if mu_multiple_uv is None:
        return np.hstack([theta, mu_single, sigma_single, mu_multiple, sigma_multiple])
    else:
        return np.hstack([theta, mu_single, sigma_single, mu_multiple, sigma_multiple, mu_multiple_uv])


def _check_params_dict(d, bounds_dict=None, fail_on_bounds=True, tolerance=0.01):
    if d is None: return d
    
    dc = {**d}
    for k in ("mu_single", "sigma_single", "mu_multiple", "sigma_multiple"):
        dc[k] = np.atleast_1d(dc[k]).flatten()[0]
        if bounds_dict is not None and k in bounds_dict:
            lower, upper = bounds_dict[k]
            if (not np.all(upper >= dc[k]) or not np.all(dc[k] >= lower)):
                if fail_on_bounds:
                    raise ValueError("bounds not met: {} = {} not within ({} {})"\
                                     .format(k, dc[k], lower, upper))
                else:
                    print("Clipping initial {} to be within bounds ({}, {}): {}"\
                        .format(k, lower, upper, dc[k]))
                    dc[k] = np.clip(dc[k], lower + tolerance, upper - tolerance)


    return dc


def nlp(params, y, L):
    return -ln_prob(y, L, *params)
