import requests
import pandas as pd
from BeautifulSoup import BeautifulSoup
import datetime
import urllib2
import numpy as np


if __name__ == "__main__":
    def date_range(start, end):
        r = (end+datetime.timedelta(days=1)-start).days
        return [start+datetime.timedelta(days=i) for i in range(r)]
    
    start = datetime.date(2011,12,1)
    end = datetime.date(2017,5,20)
    dateList = date_range(start,end)
    
    cols = ['Date', 'GameDes', 'CompType', 'Visitor', 'Home', 'AwayScore', 'HomeScore', 'OU', 'Spread']
    
    df = pd.DataFrame(columns = cols)
    
    for date in dateList:
        dateStr = date.strftime("%Y-%m-%d")
        print("Trying "+dateStr)
        try:
            page = urllib2.urlopen("http://www.covers.com/sports/NBA/matchups?selectedDate="+dateStr)
            soup = BeautifulSoup(page.read())
            games = soup.findAll("div", {"class": "cmg_matchup_game_box"})
           
            for game in games:
                if (str(game.attrMap['data-competition-type']) != 'qqq'):
                    compType = str(game.attrMap['data-competition-type'])
                    header = game.findAll("div", {"class": "cmg_matchup_header_team_names"})
                    for h in header:
                        gameDes = h.text
                    
                    
                    if ( gameDes.find(' at ') != -1 ):
                        atInd = gameDes.find(' at ')
                    elif ( gameDes.find(' vs ') != -1 ):
                        atInd = gameDes.find(' vs ')
                    try:
                        visitor = str(gameDes[:(atInd)])
                        home = str(gameDes[atInd+4:])
                    
                        date = str(game.attrMap['data-game-date'])
                        gameId = int(game.attrMap['data-event-id'])
                        try:
                            awayScore = int(game.attrMap['data-away-score'])
                        except KeyError:
                            continue
                        homeScore = int(game.attrMap['data-home-score'])
                        spread = float(game.attrMap['data-game-odd'])
                        ou = float(game.attrMap['data-game-total'])
                    
                        row = pd.DataFrame([[date, gameDes, compType, visitor, home, awayScore, homeScore, ou, spread]], columns = cols)
                        df = df.append(row)
                        print(visitor,home)
                    except ValueError:
                        continue
        except (urllib2.HTTPError & URLError) as e:
            print 'There was an error with the request'
            
    df = df.drop_duplicates()
    ooouuu = df.to_csv("OddsHistory.csv", index = False)