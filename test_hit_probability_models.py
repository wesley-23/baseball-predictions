from data.heat_chart import heat_chart
import os
import matplotlib.pyplot as plt
import math
from time import time
import multiprocessing as mp




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
            m = heat_chart(2019, to = 2022, neighbors = i, kernel = model)
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
                        loglik += math.log(max(0.01, (1 - m.get_likelihood(int(round(float(ev))), int(round(float(la)))))))
                        if m.is_hit(int(round(float(ev))), int(round(float(la)))):
                            incorrect_classifications += 1
                    else:
                        loglik += math.log(max(0.01, m.get_likelihood(int(round(float(ev))), int(round(float(la))))))
                        if not m.is_hit(int(round(float(ev))), int(round(float(la)))):
                            incorrect_classifications += 1
                    inplay_events += 1
            deviance = -2 * loglik 
            deviance /= inplay_events
            deviances.append(deviance)
            error01 = incorrect_classifications / inplay_events
            zero1.append(error01)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_vals, neighbors_d, c = 'b', label = 'nearest neighbors')
    ax1.plot(x_vals, ekernel_d, c = 'g', label = 'Epanechnikov Kernel')
    ax1.plot(x_vals, tricube_d, c = 'r', label = 'Tricube kernel')
    ax1.set_title('Test Error using Deviance vs. no. of Neighbors for Different Kernel Smoothing Methods')
    ax1.legend(loc = 'upper left')

    ax2.plot(x_vals, neighbors_01, c = 'b', label = 'nearest neighbors')
    ax2.plot(x_vals, ekerenl_01, c = 'g', label = 'Epanechnikov Kernel')
    ax2.plot(x_vals, tricube_01, c = 'r', label = 'Tricube kernel')
    ax2.set_title('Test Error using 0-1 Loss vs. no. of Neighbors for Different Kernel Smoothing Methods')
    ax2.legend(loc = 'upper left')
    fig.text(.25, .1, '*The Hit threshold was lowered to 0.45 probability for this data')
    plt.show()

def calculate_mse_statcast_xba():
    f = open('data/batter_stats/2023.csv')
    squared_error = 0
    num = 0
    next(f)
    for line in f:
        l = line.split(',')
        ba = l[7]
        xba = l[8]
        squared_error += (float(ba) - float(xba))**2
        num += 1
    print(squared_error / num)

def calculate_mse_model_xba():
    m = heat_chart(2015, to = 2022, neighbors = 1, method = 'tricube')
    squared_error = 0
    num = 0
    f = open('data/batter_stats/2023.csv', 'r')
    next(f)
    for line in f:
        l = line.split(',')
        id = int(l[2])
        path = 'data/2023_pbp/' + str(id) + '.csv'
        if not os.path.isfile(path):
            continue
        pbp = open(path, 'r')
        next(pbp)
        hits = 0
        abs = 0
        for play in pbp:
            p = play.split(',')
            ev = p[5].strip()
            la = p[6].strip()
            outcome = p[0].strip()
            if ev == 'None' or la == 'None':
                if outcome != 'Strikeout':
                    continue
            elif 'Sac' in outcome:
                continue
            elif m.is_hit(int(round(float(ev))), int(round(float(la)))):
                hits += 1
            abs += 1
        xba = hits / abs
        ba = l[7]
        squared_error += (float(ba) - float(xba)) ** 2
        num += 1
    print(squared_error / num)
        
## Tests what type of error my model is making, that is a hit misclassified as an out or an out misclassified as a hit. While the raw error rate is
## a good indication of how well my model can be used to estimate expected batting average, having a 1:1 ratio or as close to it as possible can
## "cancel out" the misclassified hits and outs in the batting average calculation. Thus, while a good model for hit probability would keep the error
## rate as low as possible, I believe that a model that misclassifies about the same number of hits as outs would also be desirable when using the 
## hit probability model to predict batting averages.

def see_error_type():
    start = time()
    m = heat_chart(2015, to = 2022,  neighbors = 50)
    # m.create_heat_chart()
    mis_as_hit = 0
    mis_as_out = 0
    f = open('data/batter_stats/2023.csv', 'r')
    next(f)
    for line in f:
        l = line.split(',')
        id = int(l[2])
        path = 'data/2023_pbp/' + str(id) + '.csv'
        if not os.path.isfile(path):
            continue
        pbp = open(path, 'r')
        next(pbp)
        hits = 0
        abs = 0
        for play in pbp:
            p = play.split(',')
            ev = p[5].strip()
            la = p[6].strip()
            outcome = p[0].strip()
            if ev == 'None' or la == 'None':
                continue
            elif 'Sac' in outcome:
                continue
            if outcome != 'Single' and outcome != 'Double' and outcome != 'Triple' and outcome != 'Home Run':
                if m.is_hit(int(round(float(ev))), int(round(float(la)))):
                    mis_as_hit += 1
            else:
                if not m.is_hit(int(round(float(ev))), int(round(float(la)))):
                    mis_as_out += 1
    print(mis_as_hit, mis_as_out)
    print(time() - start)
    


# graph_deviance_nearest_neighbors()
# compare_methods()
# calculate_mse_statcast_xba()
# calculate_mse_model_xba()
if __name__ == '__main__':
    mp.set_start_method('spawn')
    calculate_mse_model_xba()




