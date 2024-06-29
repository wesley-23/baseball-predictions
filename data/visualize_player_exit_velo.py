import matplotlib.pyplot as plt
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


    
    

create_hit_frequency_data()
