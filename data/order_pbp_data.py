import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import os
import queue
import math
import multiprocessing as mp
import ctypes as c
from time import time

"""
    Returns a dict using player ids as keys and foot speeds as the values. Data to be included in a conglomerate of all play by play data as foot
    speeds may be relevant to hit rates.\
    args:
    year(int): year of the data we are looking for
"""
def get_foot_speeds(year):
    path = 'batter_stats/' + str(year) + '.csv'
    f = open(path, 'r')
    next(f)
    sprint = {}
    for line in f:
        l = line.split(',')
        pid = int(l[2])
        try:
            s = float(l[17])
        except ValueError:
            s = 'None'
        sprint[pid] = s
    f.close()
    return sprint

# Create csv file that conglomerates all pbp data for a certain year.
def create_conglomerate_data(year):
    sprint = get_foot_speeds(year)
    path = str(year) + '_pbp_all/' + str(year) + '_pbp_conglomerate.csv'
    f = open(path, 'a')
    dir = str(year) + '_pbp'
    f.write('Outcome, Balls, Called_Strikes, Whiffs, Fouls, EV, LA, Sprint_Speed\n')
    for player in os.scandir(dir):
        p = open(player.path, 'r')
        pid = int(player.name.replace('.csv', ''))
        next(p)
        for line in p:
            line = line[:-1]
            try:
                line += ', ' + str(sprint[pid]) + '\n'
            except KeyError:
                line += ', None\n'
            f.write(line)
        p.close()
    f.close()

# Create new csv file that summarizes the amount of hits and outs that were made at each combination of exit velo rounded to the nearest int
# and launch angle
def create_hit_frequency_data(year):
    path = str(year) + '_pbp_all/' + str(year) + '_pbp_conglomerate.csv'
    dir = str(year) + '_pbp_all/'
    df = pd.read_csv(path, skipinitialspace=True)
    df = df[(df.EV != 'None') & (df.LA != 'None') & (df.Outcome != 'Catcher Interference')]
    hits = df[(df.Outcome == 'Single') | (df.Outcome == 'Double') | (df.Outcome == 'Triple') | (df.Outcome == 'Home Run')]
    outs = df[(df.Outcome != 'Single') & (df.Outcome != 'Double') & (df.Outcome != 'Triple') & (df.Outcome != 'Home Run')]
    hits_ev = round(hits['EV'].astype(float)).astype(int)
    hits_la = hits['LA'].astype(int)
    outs_ev = round(outs['EV'].astype(float)).astype(int)
    outs_la = outs['LA'].astype(int)
    entries = set()
    outcomes = {}
    for ev, la in zip(hits_ev, hits_la):
        hit = (ev, la)
        if hit in entries:
            outcome = outcomes[hit]
            outcome[0] += 1
        else:
            entries.add(hit)
            outcome = [1, 0]
            outcomes[hit] = outcome
    for ev, la in zip(outs_ev, outs_la):
        out = (ev, la)
        if out in entries:
            outcome = outcomes[out]
            outcome[1] += 1
        else:
            entries.add(out)
            outcome = [0, 1]
            outcomes[out] = outcome

    file = dir + 'EV_LA_hit_out_data.csv'
    f = open(file, 'a')
    f.write('EV, LA, Hits, Outs\n')
    for entry in entries:
        ev, la = entry
        outcome = outcomes[entry]
        hits = outcome[0]
        outs = outcome[1]
        line = str(ev) + ', ' + str(la) + ', ' + str(hits) + ', ' + str(outs) + '\n'
        f.write(line)
    f.close()



create_conglomerate_data(2015)







