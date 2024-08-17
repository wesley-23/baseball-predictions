from data.linear_methods import Logistic_Regression

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
l = Logistic_Regression(years = years)
l.fit()
print(l.predict(110, 19, 30))