import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np

# Create new csv file that summarizes the amount of hits and outs that were made at each combination of exit velo rounded to the nearest int
# and launch angle

def create_hit_frequency_data():
    df = pd.read_csv('2022_pbp_all/2022_pbp_conglomerate.csv', skipinitialspace=True)
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

    
    f = open('2022_pbp_all/EV_LA_hit_out_data.csv', 'a')
    f.write('EV, LA, Hits, Outs\n')
    for entry in entries:
        ev, la = entry
        outcome = outcomes[entry]
        hits = outcome[0]
        outs = outcome[1]
        line = str(ev) + ', ' + str(la) + ', ' + str(hits) + ', ' + str(outs) + '\n'
        f.write(line)
    f.close()

def create_hit_EV_LA_hit_chart():
    fig, ax = plt.subplots()
    df = pd.read_csv('2022_pbp_all/2022_pbp_conglomerate.csv', skipinitialspace=True)

    df = df[(df.EV != 'None') & (df.LA != 'None') & (df.Outcome != 'Catcher Interference')]
    hits = df[(df.Outcome == 'Single') | (df.Outcome == 'Double') | (df.Outcome == 'Triple') | (df.Outcome == 'Home Run')]
    outs = df[(df.Outcome != 'Single') & (df.Outcome != 'Double') & (df.Outcome != 'Triple') & (df.Outcome != 'Home Run')]

    ax.scatter(hits.EV.astype(float), hits.LA.astype(int), c = (0, 0, 1, 1), s = 1, label = 'Hits')
    ax.scatter(outs.EV.astype(float), outs.LA.astype(int), c = (1, 0, 0, 0.3), s = 1, label = 'Outs')
    ax.legend(loc = 'upper left')
    ax.set_title('Batter Outcomes from EV and LA data', fontsize = 20, pad = 20)
    plt.xlabel('Exit Velocity', fontsize = 16, labelpad= 10)
    plt.ylabel('Launch Angle', fontsize = 16, labelpad= 10)


    plt.show()

def create_hit_frequency_heat_map():
    fig, ax = plt.subplots()
    table = get_hit_frequencies_table()
    colors = [(0, 0, 0.5), (0.8, 0.9, 1)]
    cmap_name = 'heat_colors'
    cmap_ = LinearSegmentedColormap.from_list(cmap_name, colors, N = 100)

    im = ax.imshow(table, origin = 'lower', cmap = cmap_)
    fig.colorbar(im, ax = ax, label = 'Hit Rate')

    ax.set_title('2022 Hit Rate for Batted Balls From LA and EV Data\n', fontsize = 20)
    
    plt.xlabel('Exit Velocity')
    plt.ylabel('Launch Angle')
    fig.text(.25, .1,'Entries with insufficient were data estimated using nearest outcomes with similar launch angle and exit velocity. The amount of \ndata points used for each observation was at least 10.')
    plt.show()

    
def get_hit_frequencies_table():
    df = pd.read_csv('2022_pbp_all/EV_LA_hit_out_data.csv', skipinitialspace=True)
    ev = df['EV']
    la = df['LA']

    rows = la.max() + 1
    cols = ev.max() + 1
    print(la.unique)
    # print(la.min())

    table = np.zeros((rows, cols))
    hm = np.zeros((rows, cols))

    for e, l, h, o in zip(ev, la, df['Hits'], df['Outs']):
        r = h / (h + o)
        table[l][e] = r
        hm[l][e] = h + o
    # table = interpolate(table, hm, 10)
   
    return table

def interpolate(table, hm, n):
    new_table = np.zeros((len(table), len(table[0])))

    for i in range(len(table)):
        for j in range(len(table[0])):
            points = hm[i][j]
            average = table[i][j] * points
            bound = 1
            while points < n:
                for j_ in range(j - bound, j + bound + 1):
                    if i - bound >= 0:
                        if not (j_ < 0 or j_ >= len(table[0])):
                            num = hm[i - bound][j_]
                            points += num
                            average += (table[i - bound][j_] * num)
                    if i + bound < len(table):
                        if not (j_ < 0 or j_ >= len(table[0])):
                            num = hm[i + bound][j_]
                            points += num
                            average += (table[i + bound][j_] * num)
                for i_ in range(i - bound, i + bound + 1):
                    if j - bound >= 0:
                        if not (i_ < 0 or i_ >= len(table)):
                            num = hm[i_][j - bound]
                            points += num
                            average += (table[i_][j - bound] * num)
                    if j + bound < len(table[0]):
                        if not (i_ < 0 or i_ >= len(table)):
                            num = hm[i_][j + bound]
                            points += num
                            average += (table[i_][j + bound] * num)
                bound += 1
            new_table[i][j] = (average) / points
    return new_table





create_hit_frequency_heat_map()
# get_hit_frequencies_table()
# create_hit_EV_LA_hit_chart()

