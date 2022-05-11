from matplotlib import pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle
def plot_markets(markets, ax, colors=None):
    color_iter = iter(colors)
    xmin, xmax = min(m.bid for m in markets), max(m.ask for m in markets)
    xx = np.linspace(xmin*0.9, xmax*1.1, 1001)
    for q in markets:
        color = next(color_iter)
        #ax[0].vlines(0, 0, 1)
        ax.add_patch(Rectangle((q.bid, 0), q.spread(), 1/q.spread(), alpha=0.5, color=color))
        ax.plot(xx, q.gaussian().pdf(xx), color=color)
    # === Alternative way of plotting ===
    #xmin, xmax = min(m.bid for m in markets), max(m.ask for m in markets)
    #xx = np.linspace(xmin*0.9, xmax*1.1, 1001)
    #for m in markets:
    #    plt.fill_between(xx, m.uniform().pdf(xx), alpha=0.5)
    return

def plot_estimator(ax, i, est, color='k'):
    ax.hlines(i, est.bid, est.ask, color='k')
    ax.scatter(est.mid(), i, color='k')
    return