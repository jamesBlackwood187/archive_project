from pymongo import MongoClient
import pandas as pd
import numpy as np


client = MongoClient()
client = MongoClient('localhost', 27017)

db = client.bitcoin
trades = db.Trades

allTrades = trades.find()

def getTrades(client):
    db = client.bitcoin
    trades = db.Trades
    allTrades = trades.find()
    return allTrades

def getOrderBook(client):
    db = client.bitcoin
    orderBook = db.OrderBook
    book = orderBook.find()
    return book

def parseTrades(trades):
    return

def parseBook(book):
    return

if __name__ == '__main__':
    client = MongoClient()
    trades = getTrades(client)
    book = getOrderBook(client)

    for i,j in enumerate(book):
        if i < 10:
            print(j)
        if i == 10:
            break
