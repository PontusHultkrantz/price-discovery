import pandas as pd
import numpy as np
import pickle
import gzip

def _get_raw_df(ticker):
    file1 = 'data\\{}_20110301_20110331.csv'.format(ticker)
    df = pd.read_csv(file1, header=None, names=['date','time','bid','bs','ask','as'])
    df = df.dropna()
    df[df.columns] = df[df.columns].astype(float)
    
    date = pd.to_datetime(df['date'], unit='D', origin='1899-12-30', cache=True) # From Excel date format.
    seconds = pd.to_timedelta(df['time'], unit='seconds')
    df.insert(loc=0, column='timestamp', value= date + seconds)
    return df

def _get_raw_df_xbt(ticker):
    #df = analyse.load_quotes(ticker)
    #df.drop(inplace=True, columns=['symbol'])
    #df.rename(inplace=True, columns={'bidSize':'bs', 'askSize':'as', 'bidPrice':'bid', 'askPrice':'ask' })    
    #df[df.columns] = df[df.columns].astype(float)
    #with gzip.open('{}-quotes.gz'.format(ticker), "wb+") as f_hnd:
        #pickle.dump(df, f_hnd)    
    
    with gzip.open('data\\{}-quotes.gz'.format(ticker), 'rb+') as f_hnd:
        df = pickle.load(f_hnd)    

    df['time'] = df.index # expected by Microprice code
    return df

def _get_raw_df_xbt2(ticker):
    import dateutil
    src = r"data\\20210403.csv.gz"
    iter_csv = pd.read_csv(src, iterator=True, chunksize=100000, compression='gzip')
    df = pd.concat([chunk[chunk['symbol'] == 'XBTUSD'] for chunk in iter_csv])
    df.drop(inplace=True, columns=['symbol'])
    df['timestamp'] = df['timestamp'].map(dateutil.parser.isoparse) 
    df['time'] = df.index # expected by Microprice code
    #with gzip.open('{}-quotes.gz'.format(ticker), 'rb+') as f_hnd:
        #df = pickle.load(f_hnd)    

    #df['time'] = df.index # expected by Microprice code
    #return df
    return df

def _extend_fields(df):
    df['mid'] = 0.5 * (df['bid'] + df['ask'])
    df['sprd'] = 0.5 * (df['ask'] - df['bid'])
    df['imb']= df['bs'] / (df['bs'] + df['as'])
    df['wmid']= df['ask'] * df['imb'] + df['bid'] * (1-df['imb'])
    return df

def get_df(ticker):
    if ticker in ['BAC', 'CVX']:
        return _get_raw_df(ticker).pipe(_extend_fields)
    else:
        return _get_raw_df_xbt(ticker).pipe(_extend_fields)