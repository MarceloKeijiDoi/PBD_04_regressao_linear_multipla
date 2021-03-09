import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('weatherAUS.csv')

independent = dataset.iloc[:, [3, 6, 13, 15, 21, 22]].dropna()
dependent = dataset.iloc[:, -1]

ind = independent.values.reshape(-1)
dep = dependent.values.reshape(-1)

print(independent)

transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
ind = np.array((transformer.fit_transform(ind)))

ind_train, ind_test, dep_train, dep_test = train_test_split(
    ind, dep, test_size=0.2, random_state=0)

linearRegression = LinearRegression()

linearRegression.fit(ind_train, dep_train)

dep_pred = linearRegression.predict(ind_test)

np.set_printoptions(precision=2)
np.concatenate((dep_pred.reshape(len(dep_pred), 1),
                dep_test.reshape(len(dep_pred), 1)), axis=1)


for c in linearRegression.coef_:
    print(f'{c:.2f} ')
    print(f'{linearRegression.intercept_:.2f}')
