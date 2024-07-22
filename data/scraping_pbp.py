from datetime import date, timedelta
import requests
from time import sleep
import os
import multiprocessing as mp


def write_to_csv(year, line, id, lock):
    with lock:
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
    try:
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
            try:
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
            except KeyError:
                continue
        for event in away_team_offense:
            try:
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
            except KeyError:
                continue
        outcomes.append(outcome)
        return outcomes
    except (KeyError, requests.exceptions.JSONDecodeError) as error:
        raise(error)
    
    

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

def work(start_date, end_date, year, lock):
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)
    for single_date in daterange(start_date, end_date):
        url = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=' + single_date.strftime(("%Y-%m-%d"))
        with lock:
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
                    if outcome['outcome'] == None:
                        continue
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
                    write_to_csv(year, line, pid, lock)
            except (KeyError, requests.exceptions.JSONDecodeError):
                continue
            sleep(5)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    lock = mp.Lock()
    processes = []
    p1 = mp.Process(target = work, args = (date(2017, 4, 2), date(2017, 10, 3), 2017, lock))
    processes.append(p1)
    p2 = mp.Process(target = work, args = (date(2016, 4, 3), date(2016, 10, 3), 2016, lock))
    processes.append(p2)
    p3 = mp.Process(target = work, args = (date(2015, 4, 5), date(2015, 10, 5), 2015, lock))
    processes.append(p3)
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    

def fix(ids, lock, year):
    for id in ids:
            try:
                print(id)
                outcomes = scrape(id)
                print(outcomes)
                for outcome in outcomes:
                    print(outcome)
                    pid = outcome['id']
                    if outcome['outcome'] == None:
                        continue
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
                    # write_to_csv(year, line, pid, lock)
                sleep(5)
            except KeyError:
                print("fe")

   

# ids = [414086]
# lock = mp.Lock()
# fix(ids, lock, 2017)
        
        
    


