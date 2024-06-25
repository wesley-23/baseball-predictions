import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def model_ba_vs_exit_velocity():
    df = pd.read_csv('batter_stats_sorted.csv')
    training_data = pd.read_csv('batter_training_set.csv')

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
    df = pd.read_csv('batter_stats_sorted.csv')
    training = pd.read_csv('batter_training_set.csv')
    fig, ax = plt.subplots(2)
    batting_avg = df['hit'] / df['pa']
    batting_avg_training = training['next_hit']/training['next_pa']
    results = model_ba_vs_exit_velocity()

    xmin = df.min(axis = 0)['exit_velocity_avg'] - 5
    xmax = df.max(axis = 0)['exit_velocity_avg'] + 5
    ax[0].scatter(df['exit_velocity_avg'], batting_avg)
    ax[0].set_title('Batting Average Vs. Exit Velocity')
    ax[0].axline((0, results['results1'][0]), slope = results['results1'][1])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([0, 0.4])



    xmint = training.min(axis = 0)['exit_velocity_avg'] - 5
    xmaxt = training.max(axis = 0)['exit_velocity_avg'] + 5
    ax[1].scatter(training['exit_velocity_avg'], batting_avg_training)
    ax[1].set_title('Batting Average Vs. Prev Exit Velocity')
    ax[1].axline((0, results['results2'][0]), slope = results['results2'][1])
    ax[1].set_xlim([xmint, xmaxt])
    ax[1].set_ylim([0, 0.4])
    plt.show()


graph_ba_vs_exit_velocity()
# model_ba_vs_exit_velocity()

