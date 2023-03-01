# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:54:03 2023

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

# Choose relevent columns
print(df.columns)

df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'num_comp', 'hourly', 
               'employer_provided', 'jobs_state', 'same_state', 'Company_age', 'python_yn', 'excel_yn', 'aws_yn', 'spark_yn', 'job_simp', 'seniority', 'desc_len']]

# Create dummy variables
df_dum = pd.get_dummies(df_model)

# Train-Test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis = 1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression
# using statsmodels
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
print(model.fit().summary())

# using sklearn 
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm_regressor = LinearRegression()
lm_regressor.fit(X_train, y_train)


scores = cross_val_score(lm_regressor, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)
print(np.mean(scores))

# Lasso Model
l_regressor = Lasso(alpha = 0.13)
l_regressor.fit(X_train, y_train)
print(np.mean(cross_val_score(l_regressor, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)))

alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/100)
    lml = Lasso( alpha = (i/100) )
    error.append( np.mean(cross_val_score(lml, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)) )
    
plt.plot(alpha, error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
print(df_err[df_err.error == max(df_err.error)])


# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor()

print( np.mean(cross_val_score(rf_regressor, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)) )


# Tune the models GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

grid_search = GridSearchCV(rf_regressor, parameters, scoring = 'neg_mean_absolute_error', cv = 3)
grid_search.fit(X_train, y_train)

grid_search.best_score_
grid_search.best_estimator_

tpred_lm_regressor = lm_regressor.predict(X_test)
tpred_l_regressor = l_regressor.predict(X_test)
tpred_rf_regressor = grid_search.best_estimator_.predict(X_test)

print(tpred_lm_regressor[0])
print(tpred_l_regressor[0])
print(tpred_rf_regressor[0])
print(y_test[0])

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm_regressor) 
mean_absolute_error(y_test, tpred_l_regressor)
mean_absolute_error(y_test, tpred_rf_regressor)

# mean_absolute_error(y_test,(tpred_lm_regressor+tpred_rf_regressor)/2)
# Test ensembles

