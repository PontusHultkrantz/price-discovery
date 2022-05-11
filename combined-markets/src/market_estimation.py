import numpy as np
from src.market import *

def varw(markets):
    '''
    Inverse variance weights
    '''
    w = np.array([1/m.var() for m in markets])
    w = w/np.sum(w)
    return w

def _dw(obs, p=1):
    '''
    Distance weights
    https://encyclopediaofmath.org/wiki/Distance-weighted_mean
    '''
    w = np.array([np.abs(np.power(obs - x, p)).sum() for x in obs])
    if sum(w)>0: #all points same
        w = 1 / w
        w = w / np.sum(w)
    else:
        w = np.ones(len(obs))
    return w

def dw(markets):
    '''
    Distance weights
    https://encyclopediaofmath.org/wiki/Distance-weighted_mean
    '''
    obs = np.array([m.mean() for m in markets])
    return _dw(obs)


def dw_varw(markets):
    '''
    Distance weighted and variance weighted, combined.
    '''
    w = dw(markets) * varw(markets)
    return w / np.sum(w)


def online_kalman(markets):
    m_curr, v_curr = markets[0].mean(), markets[0].var()
    for i in range(1, len(markets)):
        m_new, v_new = markets[i].mean(), markets[i].var()
        w = v_new / (v_curr + v_new) # incremental weight
        m_curr = w * m_curr + (1-w) * m_new
        v_curr = w**2 * v_curr + (1-w)**2 * v_new
    
    # implied weights
    imp_weights = np.array([1/m.var() for m in markets])
    imp_weights = imp_weights / imp_weights.sum()
    #return Market.fit_moments(mean=m_curr, var=v_curr)
    return DerivedMarket(mean=m_curr, var=v_curr, src_markets=markets, src_weights=imp_weights)

def kalman(markets, rew=None):
    mus = np.array([m.mean() for m in markets])
    v = np.array([m.var() for m in markets])
    w = 1/v
    if rew is not None:
        if callable(rew):
            w *= rew(markets)
        else:
            w *= rew
    
    lamb = w / w.sum()
    mu_est = (lamb*mus).sum()
    var_est = (lamb**2*v).sum()
    #return Market.fit_moments(mean=mu_est, var=var_est)
    return DerivedMarket(mean=mu_est, var=var_est, src_markets=markets, src_weights=lamb)

def rewl(markets, est='mv', rew=None):
    '''
    rew: relevance weight, None for equal weighted (MLE), '1/v' for inverse of variance, array for custom.
    estimate: 'm' for mean, 'mv' for mean and variance, 'mmv' for mean and variance of mean.
    '''
    
    mus = np.array([m.mean() for m in markets])
    v = np.array([m.var() for m in markets])
    
    w = np.ones(len(markets))
    if rew is not None:
        if callable(rew):
            w *= rew(markets)
        else:
            w *= rew
        
    lamb = w / w.sum()
    if est in ['mv', 'mmvar']: # mean, variance
        mu_est = (lamb*mus).sum()
        var_est = (lamb*(mus-mu_est)**2).sum()
        sumlamb2 = np.sum(lamb**2)
        var_est_ub = 1/(1-sumlamb2) * var_est
        mu_est_var = sumlamb2 / (1-sumlamb2)**2 * var_est
        var_use = var_est_ub if est=='mv' else mu_est_var
        return DerivedMarket(mean=mu_est, var=var_use, src_markets=markets, src_weights=lamb)
    elif est=='m': # mean
        mu_est = np.sum(lamb/v*mus) / np.sum(lamb/v)
        mu_var_est = np.sum(lamb/v*mus) / np.sum(lamb/v)
        imp_weights = lamb/v
        imp_weights = imp_weights / imp_weights.sum()
        return DerivedMarket(mean=mu_est, var=mu_var_est, src_markets=markets, src_weights=imp_weights)
    else:
        raise ValueError("'{estimate}' is not a valid value for estimate; supported values are 'm', 'mv'")
        
        
def _dwm(obs, p=1, obs_weights=None):
    '''
    Distance Weighted Mean
    '''
    sqrdist = np.array([np.abs(np.power(obs - x, p)).sum() for x in obs])
    if sum(sqrdist)>0: #all points same
        weights = 1 / sqrdist
    else:
        weights = np.ones(len(obs))
    if obs_weights is not None:
        weights *= obs_weights
    weights = weights / weights.sum()
    est = (obs*weights).sum()
    return est

def dwm(markets, p=1, obs_weights=None):
    '''
    Distance weighted mean.
    '''
    mids = np.array([m.mid() for m in markets])
    return _dwm(mids, p, obs_weights)

def _modest(obs):
    '''
    Estimation of mode.
    https://stats.stackexchange.com/questions/176112/how-to-find-the-mode-of-a-probability-density-function
    '''
    obs = np.sort(obs)
    def rec(x):
        if len(x)==1:
            return x[0]
        elif len(x)==2:
            return np.mean(x)
        elif len(x)==3:
            d1 = x[1]-x[0]
            d2 = x[2]-x[1]
            return x[1] if d1==d2 else ((x[0]+x[1])/2 if d1<d1 else (x[1]+x[2])/2)
        else:
            h1 = int(len(x)/2)
            idx = np.argmin(x[h1:]-x[0:-h1])
            return rec(x[idx:idx+h1+1])
    return rec(obs)

import warnings
def modest(markets, dispersion=None):
    '''
    Estimation of mode.
    https://stats.stackexchange.com/questions/176112/how-to-find-the-mode-of-a-probability-density-function
    '''
    warnings.warn('Mode is estimated on the mid point, not theoretical price (known skew not considered for now).')
    mids = np.array([m.mean() for m in markets])
    mid_mode_est = _modest(mids)
    if dispersion=='bidask':
        bids = np.array([m.bid for m in markets])
        asks = np.array([m.ask for m in markets])
        warnings.warn('Note that (mode(bids)+mode(asks))/2 != mode(means)')
        mkt = Market(bid=_modest(bids), ask=_modest(asks), theo=mid_mode_est)
        return DerivedMarket(mean=mkt.mean(), var=mkt.var(), src_markets=markets, src_weights=None)
    elif dispersion=='bootstrap':
        nsim = 10000
        ests = np.empty(nsim)
        for i in range(nsim):
            sub = np.random.choice(mids, size=mids.size, replace=True, p=None)
            ests[i] = _modest(sub)
        mode_var_est = np.var(ests)
        return DerivedMarket(mean=mid_mode_est, var=mode_var_est, src_markets=markets, src_weights=None)
    else:
        raise ValueError("'{dispersion}' is not a valid value for dispersion; supported values are 'bidask', 'bootstrap'")
