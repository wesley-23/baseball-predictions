import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_exit_velo():
    df = pd.read_csv('2022_pbp/660271.csv', skipinitialspace=True)
    df = df[df.EV != 'None']
    ev = df['EV'].astype(float)
    fig, ax = plt.subplots()
    ax.set_xlim([0,125])
    ax.hist(ev, bins = 25)
    plt.show()

visualize_exit_velo()