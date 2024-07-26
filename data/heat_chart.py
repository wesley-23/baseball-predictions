import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import os
import math
import queue
import scipy.stats
import multiprocessing as mp
import ctypes as c

class heat_chart:
    MAX_LA = 90
    MAX_EV = 125
    """
    Args:
        year(int): year that the pbp data starts from
        neighbors(int): amount of observations used for nearest neighbors estimation 
        to(int): last year that pbp data is based on, if a range is desired
        kernel(string): ekernel (epanechnikov kernel)
        width(int): width of the kernel
    """
    def __init__(self, year, **kwargs):
        self.neighbors = kwargs.get('neighbors', None)
        self.year = year
        self.to = kwargs.get('to', None)
        self.kernel = kwargs.get('kernel', None)
        self.width = kwargs.get('width', None)

        if self.neighbors != None and not isinstance(self.neighbors, int):
            raise TypeError('Neighbor must be integer')
        if self.kernel != None and (self.kernel != 'ekernel' and self.kernel != 'tricube' and self.kernel != 'gaussian'):
            raise KeyError('Possible kernel methods are: ekernel (Epanchnikov Kernel), tricube, gaussian')
        
        # if self.kernel != None and not isinstance(self.width, int):
        #     raise TypeError('Width of kernel must be defined as an int if a kernel smoothing method is desired')
        
        if self.neighbors != None:
            path = ''
            if self.to is None:
                path = 'data/' + str(year) + '_pbp_all/'+str(self.neighbors) +'.csv'  
            else:
                path = 'data/' + str(year) + '_pbp_all/' + str(self.to) + '_' + str(self.neighbors) + '.csv'  
            if os.path.isfile(path):
                self.table = self.parse_file(path = path)    
            else:
                self.table = self.parse_file(n = self.neighbors)            
        else:
            path = ''
            if self.to is None:
                path = 'data/' + str(year) + '_pbp_all/EV_LA_hit_out_data.csv'
            else:
                path = 'data/' + str(year) + '_pbp_all/' + str(self.to) + '_EV_LA_hit_out_data.csv'
            if os.path.isfile(path):
                self.table = self.parse_file(path = path)
            else:
                self.table = self.parse_file()

    def parse_file(self, **kwargs):
        n = kwargs.get('n', None)
        path = kwargs.get('path', None)

        rows = 2 * self.MAX_LA + 1
        cols = self.MAX_EV
        min = self.MAX_LA

        table= np.zeros((rows, cols))
        hm = np.zeros((rows, cols))

        if path is not None:
            df = pd.read_csv(path, skipinitialspace=True)
            ev = df['EV']
            la = df['LA']
            for e, l, h, o in zip(ev, la, df['Hits'], df['Outs']):
                r = h / (h + o)
                table[l + min][e] = r
                hm[l + min][e] = h + o
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
                    old_r = table[l + min][e]
                    old_h = hm[l + min][e]
                    old_r = old_r * old_h
                    r = h + old_r
                    outcomes = o + h + old_h
                    table[l + min][e] = r / outcomes
                    hm[l + min][e] = outcomes
        if not n is None:
            if self.kernel is None:
                    table = self.interpolate(table, hm, n)
            else:
                table = self.nadaraya_watson_weighted_avg(table, hm, n)
        return table

    def interpolate_point(self, table, hm, n, i, i_end, new_table):
    # new_table = np.zeros((len(table), len(table[0])))
        for i in range(i, i_end):
            for j in range(len(table[0])):
                pq = queue.PriorityQueue()
                points = hm[i][j]
                total = points * table[i][j]
                vis = {}
                pq.put((math.sqrt((1)+ (1)), (1, 1)))
                pq.put((1, (1, 0)))
                pq.put((1, (0, 1)))
                prev = 0
                while True:
                    (curr, (dx, dy)) = pq.get()
                    # print(dx, dy)
                    if points > n and prev != curr:
                        break
                    if i - dx >= 0:
                        if not j + dy >= len(table[0]):
                            num = hm[i - dx][j + dy]
                            points += num
                            total += table[i - dx][j + dy] * num
                        if not j - dy < 0:
                            num = hm[i - dx][j - dy]
                            points += num
                            total += table[i - dx][j - dy] * num
                    if i + dx < len(table[0]):
                        if not j + dy >= len(table[0]):
                            num = hm[i + dx][j + dy]
                            points += num
                            total += table[i + dx][j + dy] * num
                        if not j - dy < 0:
                            num = hm[i + dx][j - dy]
                            points += num
                            total += table[i + dx][j - dy] * num
                    if not ((dx + 1) * 1000 + (dy + 1)) in vis:
                        pq.put((math.sqrt((dx + 1)**2 + (dy + 1)**2), (dx + 1, dy + 1)))
                        vis[(dx + 1) * 1000 + (dy + 1)] = True
                    if not ((dx + 1) * 1000 + dy) in vis:
                        pq.put((math.sqrt((dx + 1)**2 + (dy)**2), (dx + 1, dy)))
                        vis[(dx + 1) * 1000 + (dy)] = True
                    if not ((dy + 1)) in vis:
                        pq.put((math.sqrt((dx)**2 + (dy + 1)**2), (dx, dy + 1)))
                        vis[(dy + 1)] = True
                    prev = curr
                new_table[i * len(table[0]) + j] = total / points

    def interpolate(self, table, hm, n):
        new_table = np.zeros((len(table) * len(table[0])))
        shared_table = mp.Array(c.c_float, new_table) 
        processes = []
        start = 0
        for i in range(1, 9):
            end = round(len(table) * i / 8)
            p = mp.Process(target = self.interpolate_point, args = (table, hm, n, start, end, shared_table))
            processes.append(p)
            # interpolate_point(table, hm, n, start, end, new_table)
            start = end
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        new_table = np.copy(shared_table)
        output = new_table.reshape((len(table), len(table[0])))
        return output

    def nadaraya_watson_weighted_avg(self, table, hm, n):
        new_table = np.zeros((len(table), len(table[0])))
        def ekernel(x, x0, lmda):
                la, ev = x
                la0, ev0 = x0
                d = math.sqrt((la - la0)**2 + (ev - ev0)**2)
                d /= lmda
                return 3/4 * (1 - d**2) 
        def tricube(x, x0, lmda):
            la, ev = x
            la0, ev0 = x0
            d = math.sqrt((la - la0)**2 + (ev - ev0)**2)
            d /= lmda
            return (1 - (d**3))**3
        def gaussian(x, x0, lmda):
            la, ev = x
            la0, ev0 = x0
            d = math.sqrt((la - la0)**2 + (ev - ev0)**2)
            d /= lmda
            return scipy.stats.norm(0, lmda).pdf(d)

        kernel = ekernel
        if self.kernel == 'tricube':
            kernel = tricube
        if self.kernel == 'gaussian':
            kernel = gaussian


        new_table = np.zeros((len(table), len(table[0])))

        for i in range(len(table)):
            for j in range(len(table[0])):
                pq = queue.PriorityQueue()
                points = hm[i][j]
                vis = {}
                pq.put((math.sqrt((1)+ (1)), (1, 1)))
                pq.put((1, (1, 0)))
                pq.put((1, (0, 1)))
                prev = 0
                while True:
                    (curr, (dx, dy)) = pq.get()
                    if points > n and prev != curr:
                        prev = curr
                        break
                    vis[(dx) * 1000 + (dy)] = False
                    if i - dx >= 0:
                        if not j + dy >= len(table[0]):
                            num = hm[i - dx][j + dy]
                            points += num
                        if not j - dy < 0:
                            num = hm[i - dx][j - dy]
                            points += num
                    if i + dx < len(table[0]):
                        if not j + dy >= len(table[0]):
                            num = hm[i + dx][j + dy]
                            points += num
                        if not j - dy < 0:
                            num = hm[i + dx][j - dy]
                            points += num
                    if not ((dx + 1) * 1000 + (dy + 1)) in vis:
                        pq.put((math.sqrt((dx + 1)**2 + (dy + 1)**2), (dx + 1, dy + 1)))
                        vis[(dx + 1) * 1000 + (dy + 1)] = True
                    if not ((dx + 1) * 1000 + dy) in vis:
                        pq.put((math.sqrt((dx + 1)**2 + (dy)**2), (dx + 1, dy)))
                        vis[(dx + 1) * 1000 + (dy)] = True
                    if not ((dy + 1)) in vis:
                        pq.put((math.sqrt((dx)**2 + (dy + 1)**2), (dx, dy + 1)))
                        vis[(dy + 1)] = True
                    prev = curr

                if prev != 0:
                    k = kernel((i, j), (i, j), prev)
                    top = 0
                    bottom = 0
                    top += hm[i][j] * table[i][j] * k
                    bottom += k * hm[i][j]
                    for key in vis:
                        if not vis[key]:
                            dx = int(key / 1000)
                            dy = key % 1000
                            if i - dx >= 0:
                                if not j + dy >= len(table[0]):
                                    num = hm[i - dx][j + dy]
                                    k = kernel((i - dx, j + dy), (i, j), prev)
                                    top += table[i - dx][j + dy] * num * k
                                    bottom += k * num
                                if not j - dy < 0:
                                    num = hm[i - dx][j - dy]
                                    k = kernel((i - dx, j - dy), (i, j), prev)
                                    top += table[i - dx][j - dy] * num * k
                                    bottom += k * num
                            if i + dx < len(table[0]):
                                if not j + dy >= len(table[0]):
                                    num = hm[i + dx][j + dy]
                                    k = kernel((i + dx, j + dy), (i, j), prev)
                                    top += table[i + dx][j + dy] * num * k
                                    bottom += k * num
                                if not j - dy < 0:
                                    num = hm[i + dx][j - dy]
                                    k = kernel((i + dx, j - dy), (i, j), prev)
                                    top += table[i + dx][j - dy] * num * k
                                    bottom += k * num
                    new_table[i][j] = top / bottom
                else:
                    new_table[i][j] = table[i][j]
        return new_table

    
    def create_heat_chart(self):
        fig, ax = plt.subplots()
        colors = [(0, 0, 0.5), (0.8, 0.9, 1)]
        cmap_name = 'heat_colors'
        cmap_ = LinearSegmentedColormap.from_list(cmap_name, colors, N = 100)

        im = ax.imshow(self.table, origin = 'lower', cmap = cmap_, extent = (0, self.MAX_EV, -self.MAX_LA, self.MAX_LA + 1))
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
        return (self.table[index][ev]) >= .5

    def get_likelihood(self, ev, la):
        index = self.MAX_LA + la
        return (self.table[index][ev])
    
    def save(self, path):
        if not self.neighbors is None:
            if not os.path.isfile(path):
                f = open(path, 'a')
                f.write('EV, LA, Hit_Probability\n')
                for i in range(0, len(self.table)):
                    for j in range(0, len(self.table[0])):
                        line = str(j) + ', ' + str(i - self.MAX_LA) + ', ' + str(self.table[i][j]) + '\n'
                        f.write(line)
                f.close()



        

    
