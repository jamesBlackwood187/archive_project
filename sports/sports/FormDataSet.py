import requests
import pandas as pd
from BeautifulSoup import BeautifulSoup
import datetime
import time
import urllib2
import numpy as np
import json

teamMap = {'BOS':'Boston', 
            'DAL': 'Dallas', 
            'CHI' : 'Chicago', 
            'OKC': 'Oklahoma City', 
            'GSW': 'Golden State', 
            'NJN': 'Brooklyn', 
            'CHA': 'Charlotte', 
            'HOU': 'Houston', 
            'CLE': 'Cleveland',
            'DET': 'Detroit',
            'MIN': 'Minnesota', 
            'MEM': 'Memphis',
            'NOH': 'New Orleans', 
            'LAL': 'L.A. Lakers', 
            'PHI': 'Philadelphia', 
            'ATL': 'Atlanta', 
            'MIL': 'Milwaukee', 
            'POR': 'Portland',
            'IND': 'Indiana',
            'LAC': 'L.A. Clippers', 
            'DEN': 'Denver', 
            'MIA': 'Miami',
            'NYK': 'New York', 
            'SAS': 'San Antonio', 
            'ORL': 'Orlando',
            'PHX': 'Phoenix', 
            'TOR': 'Toronto',
            'SAC': 'Sacramento',
            'UTA': 'Utah',
            'BKN': 'Brooklyn',
            'NOP': 'New Orleans',
            'WAS': 'Washington'
}


def ReadOddsData(filename):
    return pd.read_csv(filename, sep = ',')

def ReadBoxScoreData(filename):
    return  pd.read_csv(filename, sep = ',')

def JoinBoxScoreOnOdd(oddsdf,bsdf):
    joinDF = pd.DataFrame()
    i = 0
    for index, row in oddsdf.iterrows():
        print i
        date = row['Date']
        boxSubset = bsdf[bsdf['Date'] == date]
        for index, row2 in boxSubset.iterrows():
            if ((row['Home'] == row2['homeTeam']) and (row['Visitor'] == row2['awayTeam'])): 
                print pd.DataFrame([row])
            elif ((row['Home'] == row2['awayTeam']) and (row['Visitor'] == row2['homeTeam'])):  
                #Fix Home Away Mismatch
                row2 = FixHomeAwayMismatch(pd.DataFrame([row2]))
        i = i + 1
        if (i%200 == 0):
            print joinDF
    return joinDF

def FixHomeAwayMismatch(row):
    columns = list(row.columns.values)
    cols = ["season","GameID", "Date","homeTeam","awayTeam",  "homeMin", "homeFGM", "homeFGA", "homeFGP", "home3PM", "home3PA", "home3PP", "homeFTM", "homeFTA", "homeFTP", "homeOREB", "homeDREB", "homeREB", "homeAST","homeSTL", "homeBLK", "homeTOV", "homePF", "homePTS", "homePTSPAINT","homePTS2NDCHANCE", "homePTSFB", "homeLARGESTLEAD", "homePTSOFFTOV" , "awayMin","awayFGM", "awayFGA", "awayFGP", "away3PM", "away3PA", "away3PP", "awayFTM", "awayFTA", "awayFTP", "awayOREB", "awayDREB", "awayREB", "awayAST", "awaySTL", "awayBLK", "awayTOV", "awayPF", "awayPTS", "awayPTSPAINT","awayPTS2NDCHANCE", "awayPTSFB", "awayLARGESTLEAD", "awayPTSOFFTOV"]
    rowR = pd.DataFrame(row,columns = cols)
    return rowR

def FormFinalDataSet(df, lookback = 30):
    return


if __name__ == '__main__':
    oddsFile = 'OddsHistory.csv'
    boxScoreFile = 'BoxScoreHistory.csv'
    oddsDF = ReadOddsData(oddsFile)
    oddsDF['Date'] = oddsDF['Date'].str[0:10]
    bsDF = ReadBoxScoreData(boxScoreFile)
    bsDF = bsDF.replace({"awayTeam":teamMap})
    bsDF = bsDF.replace({"homeTeam":teamMap})
    bsDF['GameDes'] = bsDF['awayTeam'] + " at " + bsDF['homeTeam']
    join2 = bsDF.merge(oddsDF, how = 'inner', on=["Date", "GameDes"])
    
    bsDF['GameDes'] = bsDF['homeTeam'] + " at " + bsDF['awayTeam']
    
    col_list = list(bsDF)
    col_list[3],col_list[4] = col_list[4],col_list[3]
    col_list[29:53],col_list[5:29] = col_list[5:29],col_list[29:53]
    bsDF.columns = col_list
    join3 = bsDF.merge(oddsDF, how = 'inner', on=["Date", "GameDes"])
    
    cJoin = pd.concat([join2,join3], ignore_index = True)
    cJoin = cJoin.sort(columns = "GameID")
    
    
    ooouuu = cJoin.to_csv("GameInfo.csv", index = False)
    
    