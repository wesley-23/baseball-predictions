from data.heat_chart import heat_chart
import os

"""
    Creates a csv file that contains each player's season data for a certain year augmented with my calculated expected batting average of the previous
    year's play by play data. The hit probabilities for each outcome is predicted by a model trained on play by play on previous years so that the play 
    by play data entered has not been seen by the model during training.

    Args:
        year(int): the year whose season data we are trying to predict 
"""
# def get_players(year):
#     hm = heat_chart(2021, neighbors = 20)
#     f = open('data/batter_stats/batter_stats_sorted.csv', 'r')
#     next(f)
#     for line in f:
#         line = line[:-1]
#         l = line.split(',')
#         y = int(l[3])
#         if year == y:
#             id = l[2]
#             path = 'data/' + str(2022) + '_pbp/' + id + '.csv'
#             if not os.path.isfile(path):
#                 continue
#             pbp = open(path, 'r')
#             plays = 0
#             hits = 0
#             next(pbp)
#             for play in pbp:
#                 p = play.split(',')
#                 ev = p[5].strip()
#                 la = p[6].strip()
#                 if la == 'None' or ev == 'None':
#                     if p[0].strip() != 'Strikeout':
#                         continue
#                 elif hm.is_hit(int(round(float(ev))), int(round(float(la)))):
#                     hits += 1
#                 plays += 1
#             pbp.close()
#             expected_hits = hits/plays
#             line += ','+str(expected_hits) + '\n'
#             write(line, year)

def write(line, path):
    f = open(path, 'a')
    f.write(line)
    f.close()

"""
    Creates a csv file with only season data
"""

def get_season_data(year):
    f = open('data/batter_stats/batter_stats_sorted.csv', 'r')
    path = 'data/batter_stats/' +str(year)+'.csv'
    next(f)
    for line in f:
        l = line.split(',')
        if year == int(l[3]):
            write(line, path)
    f.close()

# get_players(2023)

get_season_data(2022)