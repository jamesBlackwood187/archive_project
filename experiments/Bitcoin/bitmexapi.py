#!/usr/bin/python3.5

import websocket
import time

ws = websocket.create_connection("wss://www.bitmex.com/realtime")

message = '{"op": "subscribe", "args": ["trade:XBT"]}'
p = ws.send(message)

for i in range(3000):
    r = ws.recv()
    print(r)
    time.sleep(1)


