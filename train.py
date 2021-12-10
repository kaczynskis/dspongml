# Features: 
# Output: Paddle Location
# Using regression

import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv

pong = pd.read_csv('pong_data.csv')

print(pong.columns)
print(pong.head())

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)

y = pong['paddle_y']
x = pong[['ball_x', 'ball_y', 'ball_vx', 'ball_vy']]

model.fit(x, y)

print(model.coef_)
print(model.intercept_)

y_fit = model.predict(x)
print(y_fit)



from joblib import dump, load 
dump(model, 'mymodel.joblib') #save







