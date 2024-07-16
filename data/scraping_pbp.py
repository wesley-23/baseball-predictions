from datetime import date, timedelta
import requests
from time import sleep
import os


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 8, 30)
end_date = date(2020, 9, 28)

def write_to_csv(year, line, id):
    path = str(year) + '_pbp/' + str(id) + '.csv'
    newfile = False
    if not os.path.exists(path):
        newfile = True
    if newfile:
        f = open(path, 'a')
        header = 'Outcome, Balls, CalledStrikes, Whiffs, Fouls, EV, LA\n'
        f.write(header)
        f.write(line)
        f.close()
    else:
        f = open(path, 'a')
        f.write(line)
        f.close()

    



def scrape(gameid):
    url = 'https://baseballsavant.mlb.com/gf?game_pk=' + str(gameid)
    resp = requests.get(url)
    data = resp.json()

    home_team_offense = data['team_home']
    away_team_offense = data['team_away']
    prev_ab = 0
    outcome = {
        'id': None,
        'outcome': None,
        'balls': 0,
        'called_strikes': 0,
        'whiffs': 0,
        'fouls': 0,
        'EV': None,
        'LA': None
    }
    outcomes = []
    for event in home_team_offense:
        ab = event['ab_number']
        if (ab != prev_ab):
            if prev_ab != 0:
                outcomes.append(outcome)
            prev_ab = ab
            outcome = {
                'id': None,
                'outcome': None,
                'balls': 0,
                'called_strikes': 0,
                'whiffs': 0,
                'fouls': 0,
                'EV': None,
                'LA': None
            }
            outcome['id'] = event['batter']
            outcome['outcome'] = event['result']
        parse(outcome, event)
    for event in away_team_offense:
        ab = event['ab_number']
        if (ab != prev_ab):
            if prev_ab != 0:
                outcomes.append(outcome)
            prev_ab = ab
            outcome = {
                'id': None,
                'outcome': None,
                'balls': 0,
                'called_strikes': 0,
                'whiffs': 0,
                'fouls': 0,
                'EV': None,
                'LA': None
            }
            outcome['id'] = event['batter']
            outcome['outcome'] = event['result']
        parse(outcome, event)
    outcomes.append(outcome)
    return outcomes
    
    

def parse(outcome, event):
    if event['call'] == 'B':
            outcome['balls'] += 1
    elif event['call'] == 'S':
        if event['is_strike_swinging']:
            outcome['whiffs'] += 1
        elif event['result_code'] == 'F':
            outcome['fouls'] += 1
        else:
            outcome['called_strikes'] += 1
    elif event['call'] == 'X':
        try:
            outcome['EV'] = event['hit_speed']
        except KeyError:
            pass
        try:
            outcome['LA'] = event['hit_angle']
        except KeyError:
            pass

def main():
    for single_date in daterange(start_date, end_date):
        url = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=' + single_date.strftime(("%Y-%m-%d"))
        print(single_date.strftime(("%Y-%m-%d")))
        resp = requests.get(url)
        schedule = resp.json()
        ids = []
        for d in schedule['dates']:
            for game in d['games']:
                ids.append(game['gamePk'])
    
        list.sort(ids)
        print(ids)
        for id in ids:
            print(id)
            try:
                outcomes = scrape(id)
                for outcome in outcomes:
                    pid = outcome['id']
                    line = ''
                    line += outcome['outcome'] + ', ' + str(outcome['balls']) + ', ' + str(outcome['called_strikes']) + ", " + str(outcome['whiffs']) + ', ' + str(outcome['fouls']) + ', '
                    if outcome['EV'] != None:
                        line += outcome['EV'] + ', '
                    else:
                        line += 'None, '
                    if outcome['LA'] != None:
                        line += outcome['LA'] + '\n'
                    else:
                        line += 'None\n'
                    write_to_csv(2020, line, pid)
            except KeyError:
                continue
            sleep(5)


def fix(ids):
    for id in ids:
            print(id)
            outcomes = scrape(id)
            for outcome in outcomes:
                pid = outcome['id']
                line = ''
                line += outcome['outcome'] + ', ' + str(outcome['balls']) + ', ' + str(outcome['called_strikes']) + ", " + str(outcome['whiffs']) + ', ' + str(outcome['fouls']) + ', '
                if outcome['EV'] != None:
                    line += outcome['EV'] + ', '
                else:
                    line += 'None, '
                if outcome['LA'] != None:
                    line += outcome['LA'] + '\n'
                else:
                    line += 'None\n'
                write_to_csv(2020, line, pid)
            sleep(5)

   

main()
# ids = [631277, 631319, 631369, 631427, 631428, 631457, 631458, 631502, 631525, 631581, 631595, 631654]
# fix(ids)
        
        
    


