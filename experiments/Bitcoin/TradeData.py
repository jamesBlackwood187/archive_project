#!/usr/bin/python3.5

import websocket
import time
import pymongo
import json

ws = websocket.create_connection("wss://www.bitmex.com/realtime")

message = '{"op": "subscribe", "args": ["trade:XBT"]}'
p = ws.send(message)

client = pymongo.MongoClient()
db = client['bitcoin']
coll = db.Trades

while True:
    result = ws.recv()
    print(result)
    d = json.loads(result)
    resDB = coll.insert_one(d)

