
import pickle
import pandas
import matplotlib.pyplot as plt

def load_data(symbol, channel):

    with open(f'{symbol}-{channel}.pickle', 'rb') as f:
        objs = []
        while 1:
            try:
                objs.append(pickle.load(f))
            except EOFError:
                break
    return objs


def load_trades(symbol):
    trades = pandas.DataFrame(data=load_data(symbol, 'trades'))
    trades.set_index(pandas.DatetimeIndex(trades['timestamp']), inplace=True)
    trades.drop(['timestamp'], axis=1, inplace=True)
    return trades


def load_quotes(symbol):
    quotes = pandas.DataFrame(data=load_data(symbol, 'quotes'))
    quotes.set_index(pandas.DatetimeIndex(quotes['timestamp']), inplace=True)
    quotes.drop(['timestamp'], axis=1, inplace=True)
    return quotes


def main():

    trades = load_trades('XBTUSD')
    quotes = load_quotes('XBTUSD')

    data = pandas.concat([trades, quotes]).sort_index()
    data[['bidSize', 'bidPrice', 'askSize', 'askPrice']] = data[['bidSize', 'bidPrice', 'askSize', 'askPrice']].fillna(method='ffill')
    data['I'] = data['bidSize'].div(data['bidSize'] + data['askSize'])

    trade_data = data.loc[data['side'].notnull()]
    buy_imbalances = data['I'].loc[data['side'] == 'Buy']
    sell_imbalances = data['I'].loc[data['side'] == 'Sell']

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].hist(buy_imbalances.values, 50)
    ax[1].hist(sell_imbalances.values, 50)
    plt.show()
    print(" ")

if __name__ == '__main__':
    main()