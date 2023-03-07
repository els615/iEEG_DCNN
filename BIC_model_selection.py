import pandas as pd
import numpy as np

# take csv and make 3 dataframes
# df with all ventral and lateral id and expression taus, df with ventral, df with lateral

## DENSENET
df = pd.read_excel('/Users/emilyschwartz/Desktop/Projects/Ecog_faces_2021/semipartial_tau_updated_dense/semipartial_dense_id_exp_updated.xlsx', sheet_name='removed')
#df.columns

df_all = df[['data','Sum_i','Sum_e']]
#print(len(df_all))

df_sub = df.loc[df['Location_2'] == 'Ventral']
df_vent = df_sub[['data','Sum_i','Sum_e']]
#print(len(df_vent))

df_sub = df.loc[df['Location_2'] == 'Lateral']
df_lat = df_sub[['data','Sum_i','Sum_e']]
#len(df_lat)

# calculate bayesian information criterion for a linear regression model
from math import log
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# define BIC
# calculate bic for regression
def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

# combined electrodes 
X = df_all['Sum_i'].to_numpy()
y = df_all['Sum_e'].to_numpy()

X = X.reshape(-1, 1)

# fit model
model = LinearRegression()
model.fit(X, y)

# number of parameters
num_params = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse = mean_squared_error(y, yhat)
print('MSE: %.3f' % mse)

# calculate the bic
bic = calculate_bic(len(y), mse, num_params)
print('BIC for combined: %.3f' % bic)


# separate 
# ventral
X = df_vent['Sum_i'].to_numpy()
y = df_vent['Sum_e'].to_numpy()

X = X.reshape(-1, 1)

# fit model
model = LinearRegression()
model.fit(X, y)

# number of parameters
num_params_vent = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse_vent = mean_squared_error(y, yhat)
print('MSE for ventral: %.3f' % mse_vent)

# lateral
X = df_lat['Sum_i'].to_numpy()
y = df_lat['Sum_e'].to_numpy()

X = X.reshape(-1, 1)

# fit model
model = LinearRegression()
model.fit(X, y)

# number of parameters
num_params_lat = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse_lat = mean_squared_error(y, yhat)
print('MSE for lateral: %.3f' % mse_lat)

num_params = num_params_vent + num_params_lat

mse = (mse_vent + mse_lat)/2

# calculate the bic
bic = calculate_bic(len(y), mse, num_params)
print('BIC for separate: %.3f' % bic)

### RESNET

df = pd.read_excel('/Users/emilyschwartz/Desktop/Projects/Ecog_faces_2021/resnet18/id_exp_sp_taus.xlsx', sheet_name='Sheet3')
df.columns

df_all = df[['data','Location_2','Sum_i','Sum_e']]
#print(df_all)

#df_sub = df.loc[df['Location_2'] == 'Ventral']
df_sub = df_all.loc[df_all['Location_2'] == 1]
#print(df_sub)
df_vent = df_sub[['data','Sum_i','Sum_e']]
#print(len(df_vent))

df_sub = df_all.loc[df_all['Location_2'] == 2]
#df_sub = df.loc[df['Location_2'] == 'Lateral']
df_lat = df_sub[['data','Sum_i','Sum_e']]
#print(len(df_lat))

# calculate bayesian information criterion for a linear regression model
from math import log
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# define BIC
# calculate bic for regression
def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

# combined electrodes 
X = df_all['Sum_i'].to_numpy()
y = df_all['Sum_e'].to_numpy()

X = X.reshape(-1, 1)

# fit model
model = LinearRegression()
model.fit(X, y)

# number of parameters
num_params = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse = mean_squared_error(y, yhat)
print('MSE: %.3f' % mse)

# calculate the bic
bic = calculate_bic(len(y), mse, num_params)
print('BIC for combined: %.3f' % bic)


# separate 
# ventral
X = df_vent['Sum_i'].to_numpy()
y = df_vent['Sum_e'].to_numpy()

X = X.reshape(-1, 1)

# fit model
model = LinearRegression()
model.fit(X, y)

# number of parameters
num_params_vent = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse_vent = mean_squared_error(y, yhat)
print('MSE for ventral: %.3f' % mse_vent)

# lateral
X = df_lat['Sum_i'].to_numpy()
y = df_lat['Sum_e'].to_numpy()

X = X.reshape(-1, 1)

# fit model
model = LinearRegression()
model.fit(X, y)

# number of parameters
num_params_lat = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse_lat = mean_squared_error(y, yhat)
print('MSE for lateral: %.3f' % mse_lat)

num_params = num_params_vent + num_params_lat

mse = (mse_vent + mse_lat)/2

# calculate the bic
bic = calculate_bic(len(y), mse, num_params)
print('BIC for separate: %.3f' % bic)
