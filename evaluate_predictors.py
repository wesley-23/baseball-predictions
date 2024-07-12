import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm


def model_ba_vs_exit_velocity():
    df = pd.read_csv('data/batter_stats/batter_stats_sorted.csv')
    training_data = pd.read_csv('data/batter_stats/batter_training_set.csv')

    avg  = df['hit'] / df['pa']
    t_avg = training_data['next_hit'] / training_data['next_pa']
    X1 = pd.DataFrame({
        'intercept': np.ones(df['hit'].shape),
        'exit_velocity': df['exit_velocity_avg']
    })
    X2 = pd.DataFrame({
        'intercept': np.ones(training_data['next_hit'].shape),
        'exit_velocity': training_data['exit_velocity_avg']
    })

    model1 = sm.OLS(avg, X1)
    model2 = sm.OLS(t_avg, X2)

    results1 = model1.fit()
    results2 = model2.fit()
    output = {
        'results1': results1.params,
        'results2': results2.params
    }
    return output
    

def graph_ba_vs_exit_velocity():
    df = pd.read_csv('data/batter_stats/batter_stats_sorted.csv')
    training = pd.read_csv('data/batter_stats/batter_training_set.csv')
    fig, ax = plt.subplots(2)
    at_bats =  df['pa'] * (1 - df['bb_percent'] / 100)
    batting_avg = df['hit'] / at_bats
    batting_avg_training = training['next_hit']/training['next_pa']
    results = model_ba_vs_exit_velocity()

    xmin = df.min(axis = 0)['exit_velocity_avg'] - 5
    xmax = df.max(axis = 0)['exit_velocity_avg'] + 5
    ax[0].scatter(df['exit_velocity_avg'], batting_avg, s = 0.25)
    ax[0].set_title('Batting Average Vs. Exit Velocity')
    ax[0].axline((0, results['results1'][0]), slope = results['results1'][1])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([0, 0.4])


    xmint = training.min(axis = 0)['exit_velocity_avg'] - 5
    xmaxt = training.max(axis = 0)['exit_velocity_avg'] + 5
    ax[1].scatter(training['exit_velocity_avg'], batting_avg_training, s = 0.25)
    ax[1].set_title('Batting Average Vs. Prev Exit Velocity')
    ax[1].axline((0, results['results2'][0]), slope = results['results2'][1])
    ax[1].set_xlim([xmint, xmaxt])
    ax[1].set_ylim([0, 0.4])
    plt.show()


# Intuitively, the consistency of a player's exit velocity should correlate with a higher average. In this method, I assume that a player's exit velocity
# is normally distributed and I aim to find the variance of that distribution for each player. However, with how erratic these variances turn out and upon
# further inspection, it is clear that this assumption that a player's exit velocity is normally distributed is wrong. From the graph produced, there is
# no relationship between batting average and the standard deviation that I found.

def calculate_spread_of_exit_velocity():
    df = pd.read_csv('data/batter_stats/batter_stats_sorted.csv')
    hard_hit_z_score = norm.ppf(1 - df['hard_hit_percent'] * 0.01)
    X = 95 * np.ones(df['hard_hit_percent'].shape)
    ev_variance = ((X - df['exit_velocity_avg']) / hard_hit_z_score) ** 2
    out = pd.DataFrame({
        'z-score': hard_hit_z_score,
        'stdev': ev_variance
    })

    at_bats =  df['pa'] * (1 - df['bb_percent'] / 100)
    batting_avg = df['hit'] / at_bats

    fig, ax = plt.subplots()
    ax.scatter(out['stdev'], batting_avg)
    ax.set_title('Batting Average Vs. Player\'s EV stdev')
    plt.show()

def plot():
    df = pd.read_csv('data/batter_stats/batter_training_set.csv')
    X = df.drop(['next_pa', 'next_hit', 'next_single', 'next_double', 'next_triple', 'next_home_run', 'last_name', 'first_name', 'player_id', 'year'], axis = 1)
    batting_avg = df['next_hit'] / df['next_pa']

    fig, ax = plt.subplots(len(X.columns) + 1, len(X.columns) + 1,)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

    for i in range(0, len(X.columns) + 1):
        for j in range(0, len(X.columns) + 1):
            if i == j:
                if i < len(X.columns):
                    ax[i][j].text(0.5, 0.5, s = X.columns[i], horizontalalignment='center', verticalalignment='center', fontsize=6)
                else:
                    ax[i][j].text(0.5, 0.5, s = 'batting_avg', horizontalalignment='center', verticalalignment='center', fontsize=6)
            else:
                if i < len(X.columns) and j < len(X.columns):
                    ax[i][j].scatter(X[X.columns[j]], X[X.columns[i]], s = 0.25)
                elif i < len(X.columns):
                    ax[i][j].scatter(batting_avg, X[X.columns[i]], s = 0.25)
                elif j < len(X.columns):
                    ax[i][j].scatter(X[X.columns[j]], batting_avg, s = 0.25)
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])



    
    plt.show()

def graph_exba_vs_ba():
    df = pd.read_csv('data/training_data/2023_with_hit_predictions_K=50.csv')
    at_bats =  df['pa'] * (1 - df['bb_percent'] / 100)
    batting_avg = df['hit'] / at_bats
    exbabip = df['expected_ba'] * (at_bats)

    fig, ax = plt.subplots()
    ax.scatter(exbabip, batting_avg)
    plt.show()



# graph_ba_vs_exit_velocity()
# model_ba_vs_exit_velocity()
# calculate_spread_of_exit_velocity()
graph_exba_vs_ba()
