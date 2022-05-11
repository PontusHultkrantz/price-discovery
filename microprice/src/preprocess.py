import pandas as pd
import numpy as np

def _mirror_dynamics(df, n_imb):
    ''' Mirror state of order book and price dynamics '''
    dfm = df.copy(deep=True)
    mid0 = df['mid'].iloc[0]
    dfm[['bs', 'as']] = df[['as', 'bs']]
    dfm[['bid', 'ask']] = 2 * mid0 - df[['ask', 'bid']]
    dfm['imb'] = 1.0 - df['imb']
    dfm['imb_bucket'] = (n_imb-1) - df['imb_bucket']
    dfm['next_imb_bucket'] = (n_imb-1) - df['next_imb_bucket']
    dfm['dM'] = -df['dM']
    return dfm

def mirror(df, misc):
    # Symetrize data => 2x data (I.e. assume same dynamics for up and down moves)
    df2 = _mirror_dynamics(df, misc['n_imb'])
    df2['mirrored'] = True
    df['mirrored'] = False
    df_merged = pd.concat([df, df2], axis=0, sort=False)
    df_merged.reset_index(inplace=True, drop=True)        
    return df_merged
    

def discretize(data, n_imb, dt, n_spread):
    ''' Discretize and symmetrize data '''
    T = data.copy()
    misc = {'dt':dt, 'n_imb':n_imb, 'n_spread':n_spread}

    # ticksize:= min spread>0, 2 decimal places.
    spread = T['ask'] - T['bid']
    ticksize = np.round(min(spread.loc[spread>0]), 2)
    misc['ticksize'] = ticksize
    
    # Spread is a multiple of ticksize (discretize)
    T['spread'] = np.round((T['ask'] - T['bid']) / ticksize) * ticksize # find closest discrete tick.
    T['mid'] = 0.5 * (T['bid'] + T['ask'])
    # Filter out spreads that are outside discretized bounds. (wonky?)
    LB, UB = (0, n_spread * ticksize)
    mask = (LB < T['spread']) & (T['spread'] <= UB)
    T = T.loc[mask]
    T['imb'] = T['bs'] / (T['bs'] + T['as'])
    
    # Buckets must be symmetric around imb=0.5, due to mirroring the data and buckets later.
    low_edge = np.amin([T['imb'].min(), 1.0-T['imb'].max()])
    edges = np.linspace(low_edge, 1.0 - low_edge, n_imb + 1)
    T['imb_bucket'], bins = pd.cut(T['imb'], edges, include_lowest=True, retbins=True, labels=False)    
    misc['imb_bucket_edges'] = bins
    bucket_mid = 0.5*(bins[:-1] + bins[1::])
    misc['imb_bucket_mid'] = bucket_mid
    
    # Step ahead state variables.
    T[['next_mid', 'next_spread', 'next_time', 'next_imb_bucket', 'next_as', 'next_bs', 'next_imb']] = T[['mid', 'spread', 'time', 'imb_bucket', 'as', 'bs', 'imb']].shift(-dt)

    # Step ahead change in mid price. Mid price has half a tick resolution 0.5*(bid + ask), when either bid or ask changes.
    mid_tsize = 0.5 * ticksize
    T['dM'] = np.round((T['next_mid'] - T['mid']) / mid_tsize) * mid_tsize
    
    # Numerical reason to cover the bound by epsilon factor?
    mask = (T['dM'].abs() <= ticksize*1.1)
    T = T.loc[mask]

    return T, misc