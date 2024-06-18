import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression



def writeToTraining(s):
    f = open('batter_training_set.csv', 'a')
    f.write(s)
    f.close()

def writeToTest(s):
    f = open('batter_test_set.csv', 'a')
    f.write(s)
    f.close()

def createTrainingSets():
    f = open('batter_stats_sorted.csv', 'r')
    next(f)
    prev = ''
    for line in f:
        if not prev == '':
            prev_attrs = prev.split(',')
            curr_attrs = line.split(',')
            if curr_attrs[2] != prev_attrs[2]:
                prev = line
                continue
            if int(prev_attrs[3]) == 2019:
                prev = line
                continue
            prev_at_bats = prev_attrs[5]
            prev_attrs[5] = curr_attrs[5]
            prev_attrs[6] = curr_attrs[6]
            prev_attrs[7] = curr_attrs[7]
            prev_attrs[8] = curr_attrs[8]
            prev_attrs[9] = curr_attrs[9]
            prev_attrs[10] = curr_attrs[10]
            newLine = ''
            for attr in prev_attrs:
                newLine += attr.strip() + ','
            newLine += prev_at_bats + '\n'
            if int(curr_attrs[3]) == 2023:
                writeToTest(newLine)
                prev = line
                continue
            writeToTraining(newLine)
        prev = line


def fit_ols_model():
    df = pd.read_csv('batter_training_set.csv')
    test = pd.read_csv('batter_test_set.csv')

    Y = pd.DataFrame({
        'next_pa': df['next_pa'],
        'next_hit': df['next_hit'],
        'next_single': df['next_single'],
        'next_double': df['next_double'],
        'next_triple': df['next_triple'],
        'next_home_run': df['next_home_run']
    })
    test_X = test.drop(['next_pa', 'next_hit', 'next_single', 'next_double', 'next_triple', 'next_home_run', 'last_name', 'first_name', 'player_id', 'year'], axis = 1)
    test_X['intercept'] = np.ones(test['year'].shape)
    X = df.drop(['next_pa', 'next_hit', 'next_single', 'next_double', 'next_triple', 'next_home_run', 'last_name', 'first_name', 'player_id', 'year'], axis = 1)
    X['intercept'] = np.ones(df['year'].shape)
    regr = MultiOutputRegressor(LinearRegression()).fit(X, Y)
    pred = regr.predict(test_X.iloc[[0]])
    print(test_X.iloc[0])
    r_square = regr.score(X, Y)
    print(r_square)
    print(pred)



# createTrainingSets()
fit_ols_model()


