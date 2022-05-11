import numpy as np
from matplotlib import pyplot as plt
COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
def plot_Gstar(ticker, Gstar, misc):
    ''' Plot Micro Price Adjustment'''
    bin_edges = misc['imb_bucket_edges']
    bucket_mid = misc['imb_bucket_mid']
    
    plt.figure()
    plt.title('{} adjustments'.format(ticker))
    
    plt.plot([0, 1], [0, 0], label='Mid adj', color=COLOR_CYCLE[0])
    plt.plot([0, 1], [-0.5, 0.5], label='Weighted mid adj', color=COLOR_CYCLE[1])
    plt.hlines(-0.5, 0, 1, color='k')
    plt.hlines(0.5, 0, 1, color='k')
    #plt.fill_between([0, 1], [-0.5, -0.5], [0.5, 0.5], color='k', alpha=0.1)
    
    for i, (sprd, grp) in enumerate(Gstar.groupby(level='spread')):
        lbl = 'spread = {} tick'.format(i+1)
        plt.plot(bucket_mid, grp / sprd, label=lbl, marker='o', linewidth=0.0, color=COLOR_CYCLE[2+i])
        y_step_edges = np.append(grp.to_numpy(), grp.iloc[-1]) / sprd
        plt.step(bin_edges, y_step_edges, label='', where='post', color=COLOR_CYCLE[2+i])
    
    #plt.ylim(-0.5, 0.5)  # [-1/2, 1/2] spreads
    #plt.yticks(np.linspace(-0.5, 0.5, 10+1))
    plt.xlim([0, 1])
    plt.xticks(np.linspace(0, 1, 10+1))
    plt.legend(loc='upper left')
    plt.xlabel('Imbalance')  
    plt.ylabel('Price Adj norm. by spread')
    plt.grid()
    plt.show()
    
def plot_Bstar(B_pmf, misc):
    ''' Plot stationary transition probability mass function '''
    bin_edges = misc['imb_bucket_edges']
    bucket_mid = misc['imb_bucket_mid']

    for i, (next_sprd, marginal_pdf) in enumerate(B_pmf.groupby(level='next_spread')):
        lbl = '$s$ = {} tick'.format(i+1)
        plt.plot(bucket_mid, marginal_pdf, label=lbl, marker='o', lw=0.5, color=COLOR_CYCLE[2+i])
        plt.vlines(bucket_mid, 0, marginal_pdf, color=COLOR_CYCLE[2+i], lw=2.0)
        ticks = next_sprd / misc['ticksize']
        print('pmf(s = {:.0f} tick) = {:.2f}'.format(ticks, marginal_pdf.sum()))

    plt.xlim([0, 1])
    plt.xticks(np.linspace(0, 1, 11))

    plt.legend(loc='upper left', fontsize=11)
    plt.title('stationary transition $pmf(s, I)$')
    plt.xlabel('Imbalance $I$', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.grid()
    plt.show()    