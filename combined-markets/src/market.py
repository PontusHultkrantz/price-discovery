from scipy import stats
import numpy as np

from scipy import signal
import functools
def uniform_convolution(markets, weights=None):
    if weights is None:
        weights = np.ones(len(markets))
    lamb = weights / weights.sum()    

        # scaling causes x_i_new = x/w_i
    #xmin, xmax = min([m.bid/lam for m,lam in zip(markets, lamb)]), max([m.ask/lam for m,lam in zip(markets, lamb)])
    xmin = min(min(m.bid for m in markets), sum([m.bid*la for m,la in zip(markets,lamb)]))
    xmax = max(max(m.ask for m in markets), sum([m.ask*la for m,la in zip(markets,lamb)]))

    delta = 1e-4
    xmin, xmax = xmin-2*delta, xmax+2*delta # extend a bit to cover points on the limit
    xlim = max(xmin, xmax) # make symmetric, otherwise convolution gets offsetted, not sure how to fix.
    big_grid = np.arange(-xlim, xlim, delta)

    pmfs = [m.uniform().pdf(big_grid)*delta for m in markets]
    # if Y = w*X, then f_Y(y) = f_X(y/w)*1/w. 
    wpmfs = [m.uniform().pdf(big_grid/lam)*delta/lam for m,lam in zip(markets, lamb)]
    conv_pmf = functools.reduce(lambda pmf1,pmf2: signal.fftconvolve(pmf1,pmf2,'same'), wpmfs)

    pdfs = [f/delta for f in pmfs]
    wpdfs = [f/delta for f in wpmfs]
    conv_pdf = conv_pmf/delta
    #Integration of convoluted pdf: 
    area = np.trapz(conv_pdf, big_grid)
    assert np.abs(area-1)<0.025, "Integrated pdf does not integrate up to 1."

    # cut out grid where there is no support for any distribution
    idx = np.argwhere((xmin<=big_grid)&(big_grid<=xmax))
    big_grid = big_grid[idx]
    pmfs = [pm[idx] for pm in pmfs]
    wpmfs = [pm[idx] for pm in wpmfs]
    conv_pmf = conv_pmf[idx]
    pdfs = [pm[idx] for pm in pdfs]
    wpdfs = [pm[idx] for pm in wpdfs]
    conv_pdf = conv_pdf[idx]
    
    return big_grid, conv_pdf

class Market:
    def __init__(self, bid, ask, theo=None):
        self.bid = float(bid)
        self.ask = float(ask)
        self.theo = float(theo) if theo is not None else None
    def mid(self):
        return (self.bid + self.ask)/2
    def theo(self): #theoretical price
        return self.theo
    def skew(self):
        raise NotImplemented()
        return self.theo-self.mid
    def spread(self):
        return self.ask - self.bid
    def mean(self):
        return self.mid()
    def var(self):
        return self.spread()**2/12
    def uniform(self):
        spread = 2*(3**0.5)*self.var()**0.5
        return stats.uniform(loc=self.bid, scale=self.spread())
    def gaussian(self):
        return stats.norm(loc=self.mean(), scale=self.var()**0.5)
    @staticmethod
    def fit_moments(mean, var):
        return Market(bid=mean-3**0.5*var**0.5, ask=mean+3**0.5*var**0.5)

class DerivedMarket(Market):
    def __init__(self, mean, var, src_markets, src_weights):
        mkt = super().fit_moments(mean, var)
        super().__init__(mkt.bid, mkt.ask)
        self.src_markets = src_markets
        self.src_weights = src_weights
    def uniform_convolution(self):
        return uniform_convolution(self.src_markets, self.src_weights)