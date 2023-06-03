import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

#1 
print(df.head())

#2
prod_per_year = df.groupby("year").totalprod.mean().reset_index()

#3

X = prod_per_year.year
X = X.values.reshape(-1, 1)

#4
y= prod_per_year.totalprod

#5
plt.scatter(X,y)
plt.show()

#6+7
regr = LinearRegression()
regr.fit(X, y)

#7 slope = -88303.18915238192 // intercept = 181208083.10732976

print(regr.coef_[0])
print(regr.intercept_)

#9
y_predict = regr.predict(X)

#10
plt.plot(X,y_predict)
plt.show()

#11
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

#12
future_predict = regr.predict(X_future)

#13

plt.plot(X_future,future_predict)
plt.show()

