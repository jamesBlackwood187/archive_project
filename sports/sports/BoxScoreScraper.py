import requests
import pandas as pd
from BeautifulSoup import BeautifulSoup
import datetime
import urllib2
import numpy as np
import json
import time
from pymongo import MongoClient


def GenerateGameIDs(season):
    yearCode = season[9:11]
    gameIDs = []
    for i in range(1,2000):
        strID = str(i)
        strID = strID.rjust(5, '0')
        gameID = "002"+yearCode+strID
        gameIDs.append(gameID)
    return gameIDs


def GetBoxScore(season, gameID, database):
    headers = {
            'cookie' : "ug=55f5acec0424480a3c71ba465501c668; ugs=1; AMCVS_7FF852E2556756057F000101%40AdobeOrg=1; _gat=1; AMCV_7FF852E2556756057F000101%40AdobeOrg=817868104%7CMCIDTS%7C17130%7CMCMID%7C40832010293283617663816277025915193136%7CMCAAMLH-1480633443%7C9%7CMCAAMB-1480648494%7CNRX38WO0n5BH8Th-nqAG_A%7CMCOPTOUT-1480050894s%7CNONE%7CMCAID%7CNONE; s_fid=66526316F1B7C333-2FF678E7113817DB; s_vi=[CS]v1|2C0B2131050119DA-60000109C0045EAC[CE]; s_sq=%5B%5BB%5D%5D; s_cc=true; _ga=GA1.2.134722565.1477853785"
            ,'user-agent' : "Chrome/54.0.2840.93"
           }
           
    boxScoreURL = "http://stats.nba.com/stats/boxscoretraditionalv2?EndPeriod=10&EndRange=28800&GameID="+gameID+"&RangeType=0&"+season+"&SeasonType=Regular+Season&StartPeriod=1&StartRange=0"
    cols = ["season","GameID", "Date","awayTeam", "homeTeam", "awayMin", "awayFGM", "awayFGA", "awayFGP", "away3PM", "away3PA", "away3PP", "awayFTM", "awayFTA", "awayFTP", "awayOREB", "awayDREB", "awayREB", "awayAST", "awaySTL", "awayBLK", "awayTOV", "awayPF", "awayPTS", "awayPTSPAINT","awayPTS2NDCHANCE", "awayPTSFB", "awayLARGESTLEAD", "awayPTSOFFTOV", "homeMin", "homeFGM", "homeFGA", "homeFGP", "home3PM", "home3PA", "home3PP", "homeFTM", "homeFTA", "homeFTP", "homeOREB", "homeDREB", "homeREB", "homeAST","homeSTL", "homeBLK", "homeTOV", "homePF", "homePTS", "homePTSPAINT","homePTS2NDCHANCE", "homePTSFB", "homeLARGESTLEAD", "homePTSOFFTOV" ]
    summaryURL = "http://stats.nba.com/stats/boxscoresummaryv2?GameID="+gameID
    
    seasonY = season[-7:]
    
    
    try:
        print ("Trying "+season+" "+ gameID)
        response = requests.get(boxScoreURL, headers = headers)
        
        response2 = requests.get(summaryURL,  headers = headers)
        


        boxScore = response.json()
        summary = response2.json()
        res1 = db.gameBoxScore.insert_one(boxScore)
        res2 = db.gameSummary.insert_one(summary)


        try:
            stats = response.json()['resultSets'][1]['rowSet']
            for i,row in enumerate(stats):
                if i == 0:
                    homeTeam = str(row[3])
                    homeMin = row[5]
                    homeFGM = row[6]
                    homeFGA = row[7]
                    homeFGP = row[8]
                    home3PM = row[9]
                    home3PA = row[10]
                    home3PP = row[11]
                    homeFTM = row[12]
                    homeFTA = row[13]
                    homeFTP = row[14]
                    homeOREB = row[15]
                    homeDREB = row[16]
                    homeREB = row[17]
                    homeAST = row[18]
                    homeSTL = row[19]
                    homeBLK = row[20]
                    homeTOV = row[21]
                    homePF = row[22]
                    homePTS = row[23]
                elif i == 1:
                    awayTeam = str(row[3])
                    awayMin = row[5]
                    awayFGM = row[6]
                    awayFGA = row[7]
                    awayFGP = row[8]
                    away3PM = row[9]
                    away3PA = row[10]
                    away3PP = row[11]
                    awayFTM = row[12]
                    awayFTA = row[13]
                    awayFTP = row[14]
                    awayOREB = row[15]
                    awayDREB = row[16]
                    awayREB = row[17]
                    awayAST = row[18]
                    awaySTL = row[19]
                    awayBLK = row[20]
                    awayTOV = row[21]
                    awayPF = row[22]
                    awayPTS = row[23]
                else:
                    continue
                
            stats2 = response2.json()['resultSets'][1]['rowSet']
            date = str(response2.json()['resultSets'][0]['rowSet'][0][0][:10])
            print date
            for i,row2 in enumerate(stats2):
                if (i == 0):
                    homePTSPAINT = row2[4]
                    homePTS2NDCHANCE = row2[5]
                    homePTSFB = row2[6]
                    homeLARGESTLEAD = row2[7]
                    homePTSOFFTOV = row2[13]
                elif i == 1:
                    awayPTSPAINT = row2[4]
                    awayPTS2NDCHANCE = row2[5]
                    awayPTSFB = row2[6]
                    awayLARGESTLEAD = row2[7]
                    awayPTSOFFTOV = row2[13]
        except KeyError:
            pass
    except urllib2.HTTPError:
        print 'There was an error with the request'
    try:
        rowB = [[seasonY, gameID, date, awayTeam, homeTeam, awayMin, awayFGM, awayFGA, awayFGP, away3PM, away3PA, away3PP, awayFTM, awayFTA, awayFTP, awayOREB, awayDREB, awayREB, awayAST, awaySTL, awayBLK, awayTOV, awayPF, awayPTS, awayPTSPAINT,awayPTS2NDCHANCE, awayPTSFB, awayLARGESTLEAD, awayPTSOFFTOV, homeMin, homeFGM, homeFGA, homeFGP, home3PM, home3PA, home3PP, homeFTM, homeFTA, homeFTP, homeOREB, homeDREB, homeREB, homeAST, homeSTL, homeBLK, homeTOV, homePF, homePTS, homePTSPAINT,homePTS2NDCHANCE, homePTSFB, homeLARGESTLEAD, homePTSOFFTOV]]
        return pd.DataFrame(rowB, columns = cols)
    except UnboundLocalError:
        pass
if __name__ == "__main__":
    seasonList = ["Season=2011-12","Season=2012-13","Season=2013-14","Season=2014-15","Season=2015-16","Season=2016-17"]
    
    cols = ["season","GameID", "Date","awayTeam", "homeTeam", "awayMin", "awayFGM", "awayFGA", "awayFGP", "away3PM", "away3PA", "away3PP", "awayFTM", "awayFTA", "awayFTP", "awayOREB", "awayDREB", "awayREB", "awayAST", "awaySTL", "awayBLK", "awayTOV", "awayPF", "awayPTS", "awayPTSPAINT","awayPTS2NDCHANCE", "awayPTSFB", "awayLARGESTLEAD", "awayPTSOFFTOV", "homeMin", "homeFGM", "homeFGA", "homeFGP", "home3PM", "home3PA", "home3PP", "homeFTM", "homeFTA", "homeFTP", "homeOREB", "homeDREB", "homeREB", "homeAST","homeSTL", "homeBLK", "homeTOV", "homePF", "homePTS", "homePTSPAINT","homePTS2NDCHANCE", "homePTSFB", "homeLARGESTLEAD", "homePTSOFFTOV" ]
    df = pd.DataFrame(columns = cols)
    client = MongoClient()
    db = client.nba
    for season in seasonList:
        gameIDList = GenerateGameIDs(season)
        for gameID in gameIDList:
            try:
                  # connect to local nba datatbase
                boxScore = GetBoxScore(season, gameID, db)
                df = df.append(boxScore)
            except:
                continue
    #ooouuu = df.to_csv("BoxScoreHistory.csv", index = False)