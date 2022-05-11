from bfxapi import Client
from bfxapi.websockets.bfx_websocket import Flags

import websocket

from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels

from threading import Thread

import time
import pickle


def main():

    bfx = Client(
       #API_KEY="0aHB5lLUJph8Peh7iHbXKy006E9YuxJ3h8Oy5tbQ3WD",
       # API_SECRET="vxwt1FQr8mOOMSlKKZs3B2EqHrMKKMOf2tAwNIySiV2"
    )

    bfx = Client(
        logLevel='INFO',
        #manageOrderBooks=True
    )

    @bfx.ws.on('error')
    def log_error(err):
        print("Error: {}".format(err))

    @bfx.ws.on('order_book_update')
    def log_update(data):
        print("Book update: {}".format(data))

    @bfx.ws.on('trade_update')
    def log_update(data):
        print("Trade update: {}".format(data))

    @bfx.ws.on('order_book_snapshot')
    def log_snapshot(data):
        print("Initial book: {}".format(data))

    @bfx.ws.on('connected')
    async def start():
        await bfx.ws.enable_flag(Flags.TIMESTAMP)
        await bfx.ws.subscribe_trades('tBTCUSD')

    bfx.ws.run()


class BitmexSubscriber:

    def __init__(self):
        self.subscriptions = {}

        self._message_handlers = {'trade': self._log_trade, 'quote': self._log_quote}

    def ingest_data(self, filename, data_dict):
        with open(f'{filename}.pickle', 'ab') as f:
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

    def _log_trade(self, msg):
        #print(msg)
        for item in msg['data']:
            self.ingest_data(f"{item['symbol']}-trades", item)

    def _log_quote(self, msg):
        #print(msg)
        for item in msg['data']:
            self.ingest_data(f"{item['symbol']}-quotes", item)

    def on_message(self, msg):
        table = msg['table']
        handler = self._message_handlers[table]
        handler(msg)

    def subscribe(self, symbol, channels):

        instrument = Instrument(symbol=symbol, channels=channels)
        instrument.on('action', lambda msg: self.on_message(msg))
        thread = Thread(target=instrument.run_forever)
        self.subscriptions.update({symbol: {'channels': channels, 'thread': thread}})
        thread.start()


def bitmex_subscribe():

    websocket.enableTrace(True)

    channels = [
        InstrumentChannels.quote,
        InstrumentChannels.trade
    ]

    subscriber = BitmexSubscriber()
    subscriber.subscribe('XBTUSD', channels)


if __name__ == '__main__':
    bitmex_subscribe()