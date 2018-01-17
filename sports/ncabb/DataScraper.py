from bs4 import BeautifulSoup
import urllib2
import re
import pandas as pd
import datetime
from multiprocessing import Pool, Process

def get_games_for_date(date):
    """Returns all game url suffix for a given day"""
    month, day, year = date
    print("Getting games for " + str(month) +"/"+ str(day) + "/" +str(year))
    base_url = 'https://www.sports-reference.com/cbb/boxscores/index.cgi?'
    url_suffix = 'month='+str(month)+'&day='+str(day)+'&year='+str(year)
    full_date_url = base_url + url_suffix
    page = urllib2.urlopen(full_date_url)
    soup = BeautifulSoup(page)
    a_tags = soup.find_all('a')

    # extract all boxscorelinks
    boxscore_list = []
    for link in a_tags:
        if link.string == 'Final':
            gamelink_suffix = str(link)[9:]
            gamelink_suffix_trim, splt, tail = gamelink_suffix.partition('\"')
            boxscore_list.append(gamelink_suffix_trim)
    return boxscore_list


def score_parse(game, date):
    """Extracts information from the basic score table. Returns a dict"""
    month,day,year = date
    line_score = game.find(string=re.compile('id="line-score"'))
    soup_score = BeautifulSoup(line_score)

    teams = soup_score.find_all('a')

    # if game is with scrub non-DI team, don't bother w/ stat
    if(len(teams) < 2):
        return -1


    for i,team in enumerate(teams):
        team = str(team).partition('>')[-1][:-4]
        if i == 0:
            away_team = team
        else:
            home_team = team

    if day < 10:
        day_str = '0' + str(day)
    else:
        day_str = str(day)
    game_tag = str(away_team) + "@" + str(home_team) + str(year) + str(month) + day_str
    print(game_tag)
    for i,row in enumerate(soup_score.find_all('table')[0].find_all('tr')):
        fields = row.find_all('td')
        for j,field in enumerate(fields):
            pts = str(field).partition('>')[-1][:-5]
            if i==2:
                away_OT_pts = []
                home_OT_pts = []
                if j==1:
                    away_1H_pts = int(pts)
                elif j==2:
                    away_2H_pts = int(pts)
                elif (j > 2) and ("strong" not in str(pts)):
                    away_OT_pts.append(int(pts))
            if i==3:
                home_OT_pts =[]
                if j==1:
                    home_1H_pts = int(pts)
                elif j==2:
                    home_2H_pts = int(pts)
                elif (j > 2) and ("strong" not in str(pts)):
                    home_OT_pts.append(int(pts))
    away_team_total = away_1H_pts + away_2H_pts + sum(away_OT_pts)
    home_team_total = home_1H_pts + home_2H_pts + sum(home_OT_pts)
    return {"away_team" : away_team,
            "away_team_total": away_team_total,
            "away_1H_pts": away_1H_pts,
            "away_2H_pts": away_2H_pts,
            "away_OT_pts": away_OT_pts,
            "home_team": home_team,
            "home_team_total" : home_team_total,
            "home_1H_pts": home_1H_pts,
            "home_2H_pts": home_2H_pts,
            "home_OT_pts": home_OT_pts,
            "year": year,
            "month": month ,
            "day": day,
            "gametag" : game_tag           
            }

def four_factors_parse(game):
    """Extract stats from four factors table. Returns a dict."""
    try:
        four_factors = game.find(string=re.compile('id="four-factors"'))
        ff_soup = BeautifulSoup(four_factors)
    except TypeError:
        return {}
    for i,row in enumerate(ff_soup.find_all('table')[0].find_all('tr')):
        fields = row.find_all('td')
        if i==2:
            away_stats = [float(str(field).partition('>')[-1][:-5]) for field in fields]
        if i==3:
            home_stats = [float(str(field).partition('>')[-1][:-5]) for field in fields]
    away_ff_stat_names = ['away_pace', 'away_eFGpct', 'away_TOVpct', 'away_ORBpct', 'away_FTFGA', 'away_Ortg']
    home_ff_stat_names = ['home_pace', 'home_eFGpct', 'home_TOVpct', 'home_ORBpct', 'home_FTFGA', 'home_Ortg']

    away_ff_stats = dict(zip(away_ff_stat_names, away_stats))
    home_ff_stats = dict(zip(home_ff_stat_names, home_stats))

    ff_stats = dict(away_ff_stats, **home_ff_stats)
    return ff_stats



def basic_box_score_parse(game):
    def row_parse(team,row):
        #names, zip with td's
        return

    try:
        box_score = game.find_all("table", id = lambda s: s and s.startswith("box-score-basic"))
    except:
        return {}

    for i in [0,1]:
        rows = box_score[i].find_all('tr')
        for row in rows:
            if i ==0:
                row_parse('away', row)
            else:
                row_parse('away', row)


def advanced_box_score_parse(game):
    return


def get_boxscore(boxscore_url_suffix, date):
    """Gets Boxscore information for a single game. Returns a dict with combined stats"""
    month, day, year = date
    game_url = 'https://www.sports-reference.com'+boxscore_url_suffix
    try:
        page = urllib2.urlopen(game_url)
    except:
        return {}
    soup = BeautifulSoup(page)

    score_parsed = score_parse(soup, date)

    if score_parsed == -1:
        return {}

    four_factors_parsed = four_factors_parse(soup)

    box_score = basic_box_score_parse(soup)

    return dict(score_parsed, **four_factors_parsed)


def multi_stats_wrapper(args):
    """helper function if planning to run in parallel"""
    link, date = args
    return get_boxscore(link, date)


def get_all_boxscores(date):
    month, day, year = date
    games = get_games_for_date(date)
    
    stats_args = map(lambda x: (x,date), games)

    result_set = map(multi_stats_wrapper, stats_args)
    return result_set

def generate_dates_for_season(yr_season_end):
    season_list = [datetime.datetime(yr_season_end, 4,20) - datetime.timedelta(days = x) for x in range(180)]
    season_list = map(lambda x: (x.month, x.day, x.year), season_list)
    return season_list


def get_all_seasons():
    """Gets all seasons concurrently. Returns list of dicts with all games"""
    yr_list = range(2010, 2017)
    date_list = []
    for yr in yr_list:
        date_list.append(generate_dates_for_season(yr))
    date_list_flat = [date for sub in date_list for date in sub]

    # Can't get games concurrently. IP gets blacklisted :(
    #pool = Pool(processes = 1)
    result_set = map(get_all_boxscores, date_list_flat)
    #pool.close()
    #pool.join()
    return result_set



if __name__ == '__main__':

    h = get_all_boxscores((2,3,2014))
    df = pd.DataFrame(h)

