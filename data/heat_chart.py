import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import os

class heat_chart:
    MAX_LA = 90
    MAX_EV = 125
    """
    Args:
        year(int): year that the pbp data is from
        neighbors(int): amount of observations used for nearest neighbors estimation 
    """
    def __init__(self, year, **kwargs):
        self.neighbors = kwargs.get('neighbors', None)
        self.year = year
        self.to = kwargs.get('to', None)

        if self.neighbors != None and not isinstance(self.neighbors, int):
            raise TypeError('Neighbor must be integer')
        
        if self.neighbors != None:
            path = ''
            if self.to is None:
                path = 'data/' + str(year) + '_pbp_all/'+str(self.neighbors) +'.csv'  
            else:
                path = 'data/' + str(year) + '_pbp_all/' + str(self.to) + '_' + str(self.neighbors) + '.csv'  
            if os.path.isfile(path):
                self.hits, self.outs = self.parse_file(path = path)    
            else:
                path = 'data/' + str(year) + '_pbp_all/EV_LA_hit_out_data.csv'
                self.hits, self.outs = self.parse_file(n = self.neighbors)            
        else:
            path = ''
            if self.to is None:
                path = 'data/' + str(year) + '_pbp_all/EV_LA_hit_out_data.csv'
            else:
                path = 'data/' + str(year) + '_pbp_all/' + str(self.to) + '_EV_LA_hit_out_data.csv'
            if os.path.isfile(path):
                self.hits, self.outs = self.parse_file(path = path)
            else:
                self.hits, self.outs = self.parse_file()

    def parse_file(self, **kwargs):
        n = kwargs.get('n', None)
        path = kwargs.get('path', None)

        rows = 2 * self.MAX_LA + 1
        cols = self.MAX_EV
        min = self.MAX_LA

        hits = np.zeros((rows, cols))
        outs = np.zeros((rows, cols))

        if path is not None:
            df = pd.read_csv(path, skipinitialspace=True)
            ev = df['EV']
            la = df['LA']
            for e, l, h, o in zip(ev, la, df['Hits'], df['Outs']):
                hits[l + min][e] = h
                outs[l + min][e] = o
        else:
            end = self.year + 1
            if self.to is not None:
                end = self.to + 1
            for i in range(self.year, end):
                path = 'data/' + str(i) + '_pbp_all/EV_LA_hit_out_data.csv'
                df = pd.read_csv(path, skipinitialspace=True)
                ev = df['EV']
                la = df['LA']
                for e, l, h, o in zip(ev, la, df['Hits'], df['Outs']):
                    hits[l + min][e] = h
                    outs[l + min][e] = o
        if not n is None:
            hits, outs = self.interpolate(hits, outs, n)
        return (hits, outs)

    def interpolate(self, hits, outs, n):
        new_hits = np.zeros((len(hits), len(hits[0])))
        new_outs = np.zeros((len(hits), len(hits[0])))
        for i in range(len(hits)):
            for j in range(len(hits[0])):
                points = hits[i][j] + outs[i][j]
                average = hits[i][j]
                bound = 1
                while points < n:
                    for j_ in range(j - bound, j + bound + 1):
                        if i - bound >= 0:
                            if not (j_ < 0 or j_ >= len(hits[0])):
                                points += (hits[i - bound][j_] + outs[i - bound][j_])
                                average += (hits[i - bound][j_])
                        if i + bound < len(hits):
                            if not (j_ < 0 or j_ >= len(hits[0])):
                                points += (hits[i + bound][j_] + outs[i + bound][j_])
                                average += (hits[i + bound][j_])
                    for i_ in range(i - bound, i + bound + 1):
                        if j - bound >= 0:
                            if not (i_ < 0 or i_ >= len(hits)):
                                points += (hits[i_][j - bound] + outs[i_][j - bound])
                                average += (hits[i_][j - bound])
                        if j + bound < len(hits[0]):
                            if not (i_ < 0 or i_ >= len(hits)):
                                points += (hits[i_][j + bound] + outs[i_][j + bound])
                                average += (hits[i_][j + bound])
                    bound += 1
                new_hits[i][j] = average
                new_outs[i][j] = points - average
        return (new_hits, new_outs)
    
    def create_heat_chart(self):
        fig, ax = plt.subplots()
        colors = [(0, 0, 0.5), (0.8, 0.9, 1)]
        cmap_name = 'heat_colors'
        cmap_ = LinearSegmentedColormap.from_list(cmap_name, colors, N = 100)

        table = self.hits / (self.hits + self.outs)
        im = ax.imshow(table, origin = 'lower', cmap = cmap_, extent = (0, self.MAX_EV, -self.MAX_LA, self.MAX_LA + 1))
        fig.colorbar(im, ax = ax, label = 'Hit Rate')

        year_title = ''
        if self.to is not None:
            year_title = str(self.year) + '-' + str(self.to)
        chart_title = year_title + ' Hit Rate for Batted Balls from LA and EV Data\n'
        ax.set_title(chart_title, fontsize = 20)

        plt.xlabel('Exit Velocity')
        plt.ylabel('Launch Angle')

        chart_desc = ''
        if self.neighbors != None:
            chart_desc = 'Entries with insufficient were data estimated using nearest outcomes with similar launch angle and exit velocity. The amount of \ndata points used for each observation was at least ' + str(self.neighbors) + '.'
        else:
            chart_desc = 'Hit frequencies from LA and EV data from ' + year_title + '. No means of estimation were used for entries with insufficient data. Entries with no\ndata were assigned the value 0.'
        fig.text(.25, .1, chart_desc)
        plt.show()

    def is_hit(self, ev, la):
        index = self.MAX_LA + la
        hits = self.hits[index][ev]
        total = hits + self.outs[index][ev]
        return (hits / total) >= .5

    def __del__(self):
        if not self.neighbors is None:
            path = ''
            if self.to is None:
                path = 'data/' + str(self.year) + '_pbp_all/'+str(self.neighbors) +'.csv'    
            else:
                path = 'data/' + str(self.year) + '_pbp_all/' + str(self.to) + '_' + str(self.neighbors) + '.csv' 
            if not os.path.isfile(path):
                f = open(path, 'a')
                f.write('EV, LA, Hits, Outs\n')
                for i in range(0, len(self.hits)):
                    for j in range(0, len(self.hits[0])):
                        line = str(j) + ', ' + str(i - self.MAX_LA) + ', ' + str(self.hits[i][j]) + ', ' + str(self.outs[i][j]) + '\n'
                        f.write(line)
                f.close()



        

    
