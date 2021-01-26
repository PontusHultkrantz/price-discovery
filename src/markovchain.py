import pandas as pd
import numpy as np

def _full_stspace(T):
    ''' Get all visited unique states. '''
    def value_union(columns):
        return np.sort(np.unique(T[columns].drop_duplicates().to_numpy().flatten()))

    spread_states = value_union(columns=['spread', 'next_spread'])
    imb_states = value_union(columns=['imb_bucket', 'next_imb_bucket'])
    dM_states = np.sort(T['dM'].unique())
    return {'spread':spread_states, 'imb_bucket':imb_states, 'dM':dM_states}

def _stspace_pd_index(stspace):
    ''' Create Markov chain matrix DataFrame indices.'''
    spr_imb = pd.MultiIndex.from_product([stspace['spread'], stspace['imb_bucket']], names=['spread', 'imb_bucket'])
    nxt_spr_imb = spr_imb.copy().set_names(names=['next_spread', 'next_imb_bucket'])
    dM = pd.Index(stspace['dM'], name='dM')
    return {'spr_imb':spr_imb, 'nxt_spr_imb':nxt_spr_imb, 'dM':dM}

def estimate(T):
    st_idx = _stspace_pd_index(_full_stspace(T))

    # === Transition matrix {spread, imb_bucket} → {next_spread, next_imb_bucket, mid_change} ===
    # Get counts
    trans = T.pivot_table(index=['spread', 'imb_bucket'], 
                     columns=['next_spread', 'next_imb_bucket', 'dM'], 
                     fill_value=0,
                     aggfunc='size')
    
    # Get mean MLE estimates
    trans = trans.divide(trans.sum(axis=1), axis=0)

    # Mask for absorbed transitions (mid price change occurs)
    absorb = trans.columns.get_level_values('dM') != 0.0
    
    # No absorb transitions, collapse (sum) mid change states {spread, imb_bucket} → {next_spread, next_imb_bucket, dM==0}.
    Q = trans.loc[:, ~absorb].sum(axis=1, level=['next_spread', 'next_imb_bucket'])
    Q = Q.reindex(index=st_idx['spr_imb'], columns=st_idx['nxt_spr_imb'], fill_value=0.0)

    # Absorb transitions, collapse (sum) all but mid_change states {spread, imb_bucket} → {dM!=0}.
    R1 = trans.loc[:, absorb].sum(axis=1, level=['dM'])
    R1 = R1.reindex(index=st_idx['spr_imb'], columns=st_idx['dM'], fill_value=0.0)
    SHOW_K0_STATE = False # Whether to show K=0 as a state (cosmetic)
    if not SHOW_K0_STATE:
        R1 = R1.loc[:, ~np.isclose(R1.columns.get_level_values('dM'), 0.0)]

    # Absorb transitions, collapse (sum) all next_spreads and next_imb into one per dM {spread, imb_bucket} → {next_spread, next_imb_bucket, dM!=0}.
    R2 = trans.loc[:, absorb].sum(axis=1, level=['next_spread', 'next_imb_bucket'])
    R2 = R2.reindex(index=st_idx['spr_imb'], columns=st_idx['nxt_spr_imb'], fill_value=0.0)

    # Jump sizes
    K = st_idx['dM'].to_numpy()
    if not SHOW_K0_STATE:
        K = K[K!=0.0]
    eye = np.eye(Q.shape[0])
    G1 = np.linalg.inv(eye - Q) @ R1 @ K # inv() converts DataFrame into ndarray, so pandas index is gone here.
    G1.index = Q.index # so we set it back.
    B = np.linalg.inv(eye - Q) @ R2
    B.index = Q.index
    Q2 = Q.copy()
    
    return G1, B, Q, Q2, R1, R2, K


def _transition_asymtotics(G1, B):
    '''
    P_k = M + sum_0^{k-1}(B^i) * G1
    Gstar := lim(P_k - M)
    '''
    vv, ee = np.linalg.eig(B)
    unit_idx = np.argmax(vv)
    Bstar = np.real(np.outer(ee[:, unit_idx], np.linalg.inv(ee)[unit_idx]))
    Bstar = pd.DataFrame(data=Bstar, index=B.index, columns=B.columns)
    vv[unit_idx] = 0.0
    vvgeosum = vv/(1 - vv)
    #vvgeosum = (vv - vv**(order))/(1 - vv)
    Gsum = ee @ np.diag(vvgeosum) @ np.linalg.inv(ee)
    Gstar = (G1 + Gsum @ G1).to_numpy().real
    Gstar = pd.Series(data=Gstar, index=G1.index)
    return Gstar, Bstar

def calc_price_adj(G1, B, order='stationary'):
    ''' Calculate Price Adjustments.'''
    if order == 'stationary':
        return _transition_asymtotics(G1, B)
    elif isinstance(order, int): # Crude method,.. if num issues, use the eigendecomp for non asymtotic.
        Bstar = pd.DataFrame(data=mpow(B, order), index=B.index, columns=B.columns)
        Gstar = sum(mpow(B, i) for i in range(order)) @ G1
        Gstar = pd.Series(data=Gstar, index=B.index)
        return Gstar, Bstar
    else:
        raise ValueError()