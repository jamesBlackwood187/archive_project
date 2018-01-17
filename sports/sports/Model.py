import requests
import pandas as pd
from BeautifulSoup import BeautifulSoup
import datetime
import time
import urllib2
import numpy as np
import json


def ReadGameInfo(filename):
    return pd.read_csv(filename, sep = ',')

def InputsFromPastN(subdf,team):
    awayStats = subdf[subdf['awayTeam'] == team]
    homeStats = subdf[subdf['homeTeam'] == team]
    
    
    awayMin           = awayStats['awayMin'].str[:3]
    awayMin           = awayMin.astype(int)
    awayMin           = sum(awayMin)
    awayPtsGiven      = awayStats['homePTS'].sum() 
    awayFGMGiven      = awayStats['homeFGM'].sum()
    awayFGAGiven      = awayStats['homeFGA'].sum()
    away3PMGiven      = awayStats['home3PM'].sum()
    away3PAGiven      = awayStats['home3PA'].sum()
    awayFTMGiven      = awayStats['homeFTM'].sum()
    awayFTAGiven      = awayStats['homeFTA'].sum()
    
    awayTOVGiven           = awayStats['homeTOV'].sum()
    awayREBGiven           = awayStats['homeREB'].sum()
    awayOREBGiven          = awayStats['homeREB'].sum()
    away2NDCHANCEGiven     = awayStats['homePTS2NDCHANCE'].sum()
    awayASTGiven           = awayStats['homeAST'].sum()
    awayLARGESTLEADGiven   = awayStats['homeLARGESTLEAD'].sum()
    
    
    awayPTSOFFTOVGiven= awayStats['homePTSOFFTOV'].sum()
    awayFBPtsGiven    = awayStats['homePTSFB'].sum()
    awayPtsPaintGiven = awayStats['homePTSPAINT'].sum()
    
    awayPts           = awayStats['awayPTS'].sum()
    awayFGM           = awayStats['awayFGM'].sum()
    awayFGA           = awayStats['awayFGA'].sum()
    away3PM           = awayStats['away3PM'].sum()
    away3PA           = awayStats['away3PA'].sum()
    awayFTM           = awayStats['awayFTM'].sum()
    awayFTA           = awayStats['awayFTA'].sum()
    
    awayTOV           = awayStats['awayTOV'].sum()
    awayREB           = awayStats['awayREB'].sum()
    awayOREB          = awayStats['awayREB'].sum()
    away2NDCHANCE     = awayStats['awayPTS2NDCHANCE'].sum()
    awayAST           = awayStats['awayAST'].sum()
    awayLARGESTLEAD   = awayStats['awayLARGESTLEAD'].sum()
    
    awayPTSOFFTOV     = awayStats['awayPTSOFFTOV'].sum()
    awayFBPts         = awayStats['awayPTSFB'].sum()
    awayPtsPaint      = awayStats['awayPTSPAINT'].sum()
    

    homeMin           = homeStats['homeMin'].str[:3]
    homeMin           = homeMin.astype(int)
    homeMin           = sum(homeMin)
    homePtsGiven      = homeStats['awayPTS'].sum() 
    homeFGMGiven      = homeStats['awayFGM'].sum()
    homeFGAGiven      = homeStats['awayFGA'].sum()
    home3PMGiven      = homeStats['away3PM'].sum()
    home3PAGiven      = homeStats['away3PA'].sum()
    homeFTMGiven      = homeStats['awayFTM'].sum()
    homeFTAGiven      = homeStats['awayFTA'].sum()
    
    homeTOVGiven           = homeStats['awayTOV'].sum()
    homeREBGiven           = homeStats['awayREB'].sum()
    homeOREBGiven          = homeStats['awayREB'].sum()
    home2NDCHANCEGiven     = homeStats['awayPTS2NDCHANCE'].sum()
    homeASTGiven           = homeStats['awayAST'].sum()
    homeLARGESTLEADGiven   = homeStats['awayLARGESTLEAD'].sum()
    
    
    homePTSOFFTOVGiven= homeStats['awayPTSOFFTOV'].sum()
    homeFBPtsGiven    = homeStats['awayPTSFB'].sum()
    homePtsPaintGiven = homeStats['awayPTSPAINT'].sum()
    
    
    homePts           = homeStats['awayPTS'].sum()
    homeFGM           = homeStats['homeFGM'].sum()
    homeFGA           = homeStats['homeFGA'].sum()
    home3PM           = homeStats['home3PM'].sum()
    home3PA           = homeStats['home3PA'].sum()
    homeFTM           = homeStats['homeFTM'].sum()
    homeFTA           = homeStats['homeFTA'].sum()
    
    homeTOV           = homeStats['homeTOV'].sum()
    homeREB           = homeStats['homeREB'].sum()
    homeOREB          = homeStats['homeREB'].sum()
    home2NDCHANCE     = homeStats['homePTS2NDCHANCE'].sum()
    homeAST           = homeStats['homeAST'].sum()
    homeLARGESTLEAD   = homeStats['homeLARGESTLEAD'].sum()
    
    homePTSOFFTOV     = homeStats['homePTSOFFTOV'].sum()
    homeFBPts         = homeStats['homePTSFB'].sum()
    homePtsPaint      = homeStats['homePTSPAINT'].sum()
    
    totalMin = homeMin + awayMin
    
    inputs = (1.0 / totalMin) * np.array([homePtsGiven+awayPtsGiven, homeFGMGiven+awayFGMGiven, homeFGAGiven+awayFGAGiven, home3PMGiven+away3PMGiven, home3PAGiven+away3PAGiven, homeFTMGiven+awayFTMGiven, homeFTAGiven+awayFTAGiven, homeTOVGiven+awayTOVGiven,homeASTGiven+awayASTGiven, homeREBGiven+awayREBGiven, homeOREBGiven+awayOREBGiven,home2NDCHANCEGiven+away2NDCHANCEGiven, homeLARGESTLEADGiven+awayLARGESTLEADGiven, homePTSOFFTOVGiven+awayPTSOFFTOVGiven, homeFBPtsGiven+awayFBPtsGiven, homePtsPaintGiven+awayPtsPaintGiven, homePts+awayPts, homeFGM+awayFGM, homeFGA+awayFGA, home3PM+away3PM, home3PA+away3PA, homeFTM+awayFTM, homeFTA+awayFTA,  homeTOV+awayTOV,homeAST+awayAST, homeREB+awayREB, homeOREB+awayOREB,home2NDCHANCE+away2NDCHANCE, homeLARGESTLEAD+awayLARGESTLEAD, homePTSOFFTOV+awayPTSOFFTOV, homeFBPts+awayFBPts, homePtsPaint+awayPtsPaint])
    return inputs
    

def ComputeAwayInputs(df,gameid, season, awayteam, lookbackN):
    dfSub = df[((df['homeTeam'] == awayteam) | (df['awayTeam'] == awayteam)) & (df["GameID"] < gameid) & (df['season'] == season)][-lookbackN:]
    if len(dfSub) < 10:
        return np.array([])
    else:
        a = InputsFromPastN(dfSub,awayteam)
        return a

def ComputeHomeInputs(df,gameid,season, hometeam, lookbackN):
    dfSub = df[((df['homeTeam'] == hometeam) | (df['awayTeam'] == hometeam)) & (df["GameID"] < gameid) & (df['season'] == season)][-lookbackN:]
    if len(dfSub) < 10:
        return np.array([])
    else:
        a = InputsFromPastN(dfSub,hometeam)
        return a
        

def FormTrainData(gameInfo, lookback = 20):
    dataFrame = pd.DataFrame()
    for i in gameInfo.index:
        print i
        row = gameInfo.ix[i]
        gameID = row.GameID
        season = row["season"]
        homeTeam = row["homeTeam"]
        awayTeam = row["awayTeam"]
        ou = row["OU"]
        spread = row["Spread"]
        target = row["awayPTS"] + row["homePTS"] - ou
        result = (row["homePTS"] - row["awayPTS"]) + spread
        awayInputs = ComputeAwayInputs(gameInfo, gameID, season, awayTeam,lookback)
        homeInputs = ComputeHomeInputs(gameInfo,gameID, season, homeTeam,lookback)
        inputs = np.concatenate([awayInputs,homeInputs])
        if (len(inputs) ==0):
            continue
        else:
            tableRow = np.concatenate([np.array([target]),np.array([ou]),np.array([spread]),inputs])
            dataFrame = dataFrame.append([tableRow])
    return dataFrame

if __name__ == "__main__":
    gameFile = "GameInfo.csv"
    gameInfo = ReadGameInfo(gameFile)
    
    finDataSet = FormTrainData(gameInfo, 25)
    finDataSet = finDataSet.dropna()
    ooouuu = finDataSet.to_csv("dataSet.csv", index=False)