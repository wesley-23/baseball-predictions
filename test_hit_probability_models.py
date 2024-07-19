from data.heat_chart import heat_chart
import os
import matplotlib.pyplot as plt




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

def compare_methods():
    neighbors_d = []
    ekernel_d= []
    tricube_d = []
    neighbors_01 = []
    ekerenl_01 = []
    tricube_01 = []
    x_vals = []

    for i in range(1, 151, 5):
        x_vals.append(i)
        models = [None, 'ekernel', 'tricube']
        lists_d = [neighbors_d, ekernel_d, tricube_d]
        lists_01 = [neighbors_01, ekerenl_01, tricube_01]
        print(i)
        for model, deviances, zero1 in zip(models, lists_d, lists_01):
            m = heat_chart(2019, neighbors = i, to = 2022, kernel = model)
            deviance = 0
            loglik = 0
            inplay_events = 0
            incorrect_classifications = 0
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
                        if m.is_hit(int(round(float(ev))), int(round(float(la)))):
                            incorrect_classifications += 1
                    else:
                        loglik += m.get_likelihood(int(round(float(ev))), int(round(float(la))))
                        if not m.is_hit(int(round(float(ev))), int(round(float(la)))):
                            incorrect_classifications += 1
                    inplay_events += 1
            deviance = -2 * loglik 
            deviances.append(deviance)
            error01 = incorrect_classifications / inplay_events
            zero1.append(error01)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_vals, neighbors_d, c = 'b', label = 'nearest neighbors')
    ax1.plot(x_vals, ekernel_d, c = 'g', label = 'Epanechnikov Kernel')
    ax1.plot(x_vals, tricube_d, c = 'r', label = 'Tricube kernel')
    ax1.set_title('Deviances vs. no. of Neighbors for Different Kernel Smoothing Methods')
    ax1.legend(loc = 'upper left')

    ax2.plot(x_vals, neighbors_01, c = 'b', label = 'nearest neighbors')
    ax2.plot(x_vals, ekerenl_01, c = 'g', label = 'Epanchnikov Kernel')
    ax2.plot(x_vals, tricube_01, c = 'r', label = 'Tricube kernel')
    ax2.set_title('0-1 Error vs. no. of Neighbors for Different Kernel Smoothing Methods')
    ax2.legend(loc = 'upper left')
    plt.show()

# graph_deviance_nearest_neighbors()
compare_methods()

        
                       



