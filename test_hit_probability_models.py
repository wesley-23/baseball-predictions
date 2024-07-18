from data.heat_chart import heat_chart
import os
import matplotlib.pyplot as plt


def find_best_hit_model():
    best = None
    pass

    ## Nearest Neighbors


def graph_deviance_nearest_neighbors():
    neighbors = []
    deviances = []
    for i in range(1, 150):
        neighbors.append(i)
        print(i)
        m = heat_chart(2019, neighbors = i, to = 2022)
        deviance = 0
        loglik = 0
        f = open('data/batter_stats/2023.csv')
        next(f)
        for line in f:
            l = line.split(',')
            id = l[2]
            path = 'data/' + str(2022) + '_pbp/' + id + '.csv'
            if not os.path.isfile(path):
                continue
            pbp = open(path, 'r')
            next(pbp)
            for play in pbp:
                p = play.split(',')
                ev = p[5].strip()
                la = p[6].strip()
                outcome = p[0].strip()
                if ev == 'None' or la == 'None':
                    continue
                if outcome != 'Single' and outcome != 'Double' and outcome != 'Triple' and outcome != 'Home Run':
                    loglik += (1 - m.get_likelihood(int(round(float(ev))), int(round(float(la)))))
                else:
                    loglik += m.get_likelihood(int(round(float(ev))), int(round(float(la))))
        deviance = -2 * loglik 
        deviances.append(deviance)
    plt.plot(neighbors, deviances, '-o')
    plt.show()

graph_deviance_nearest_neighbors()

        
                       



