## Fix scraping error where all star game data was included with player's season data

import requests
from scraping_pbp import scrape


"""
    Method that will search for all star game play by play data in my data sets. Returns true if all of the data is present
    args:
        pk(int): the primary key of the all star game we are searching.
"""
def all_star_game_was_scraped(pk, year):
    contains = True
    outcomes = scrape(pk)
    for outcome in outcomes:
        pid = outcome['id']
        print(outcome)
        outcome_contained = False
        path = str(year) + '_pbp/' + str(pid) + '.csv'
        f = open(path, 'r')
        for line in f:
            l = line.split(',')
            if (l[0].strip() == outcome['outcome'] and int(l[1].strip()) == outcome['balls'] and int(l[2].strip()) == outcome['called_strikes']
                and int(l[3].strip()) == outcome['whiffs'] and int(l[4].strip()) == outcome['fouls']):
                if (l[5].strip() == outcome['EV'] and l[6].strip() == outcome['LA']):
                    outcome_contained = True
                    break
                if (outcome['EV'] == None and outcome['LA'] == None and l[5].strip() == 'None' and l[6].strip() == 'None'):
                    outcome_contained = True
                    break
                if (outcome['EV'] == None and l[5].strip == 'None' and l[6].strip() == outcome['LA']):
                    outcome_contained = True
                    break
                if (outcome['LA'] == None and l[6].strip() == 'None' and l[5].strip() == outcome['EV']):
                    outcome_contained = True
                    break
        if not outcome_contained:
            contains = False

            
    return contains

"""
    Onced verified that we counted the all star game data, use this method to remove the data
    args:
        pk(int): game id
        year(int): year of all star game
"""
def delete_all_star_data(pk, year):
    outcomes = scrape(pk)
    for outcome in outcomes:
        pid = outcome['id']
        path = str(year) + '_pbp/' + str(pid) + '.csv'
        f = open(path, 'r')
        deleted = False

        lines = ''
        for line in f:
            l = line.split(',')
            if not deleted:
                if (l[0].strip() == outcome['outcome'] and int(l[1].strip()) == outcome['balls'] and int(l[2].strip()) == outcome['called_strikes']
                    and int(l[3].strip()) == outcome['whiffs'] and int(l[4].strip()) == outcome['fouls']):
                    if (l[5].strip() == outcome['EV'] and l[6].strip() == outcome['LA']):
                        print(pid)
                        deleted = True
                        continue
                    if (outcome['EV'] == None and outcome['LA'] == None and l[5].strip() == 'None' and l[6].strip() == 'None'):
                        print(pid)
                        deleted = True
                        continue
                    if (outcome['EV'] == None and l[5].strip == 'None' and l[6].strip() == outcome['LA']):
                        print(pid)
                        deleted = True
                        continue
                    if (outcome['LA'] == None and l[6].strip() == 'None' and l[5].strip() == outcome['EV']):
                        print(pid)
                        deleted = True
                        continue
            lines += line
        f.close()
        w = open(path, 'w')
        w.write(lines)
        w.close()






# print(all_star_game_was_scraped(663466, 2022))
delete_all_star_data(663466, 2022)