import numpy as np
import pandas as pd
import seaborn as sns
import joblib as joblib
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.api import VAR
from evaluation_metric import evaluate, mean_absolute_percentage_error
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
import xgboost as xgb
from hyperopt import hp, tpe, fmin
from hyperopt import Trials
from hyperopt import STATUS_OK
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

savedir_models = Path('Models/').resolve()
loaddir = Path('Data/').resolve()
nobs = 144 # 6 days
lookback_window = 24*180 # 90 days
columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']

# %% ------------------------------- Read data ------------------------------- #

pickle_path = Path('Data/serialized/processed_x_train_scaled_pickle').resolve()
train = pd.read_pickle(pickle_path)
train = train.dropna()

pickle_path = Path('Data/serialized/processed_x_test_scaled_pickle').resolve()
test = pd.read_pickle(pickle_path)

observed = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
test_true = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)

X_train = train.tail(train.shape[0]-nobs).drop(['grid1-load', 'grid1-loss', 'grid1-temp', 'grid3-load', 'grid3-loss', 'grid3-temp'], axis=1)
y_train = train['grid2-loss'].shift(nobs).dropna()

frames = [train.tail(lookback_window+nobs), test]
df_test = pd.concat(frames)

X_test = pd.concat(frames).head(test.shape[0]).drop(['grid1-load', 'grid1-loss', 'grid1-temp', 'grid3-load', 'grid3-loss', 'grid3-temp'], axis=1)
y_test = test['grid2-loss']

y_true = np.array(test_true['grid2-loss'])
y_observed = np.array(observed['grid2-loss'][10000:])

scaler_grid1 = joblib.load(savedir_models / 'scaler_grid1.sav')
scaler_grid2 = joblib.load(savedir_models / 'scaler_grid2.sav')
scaler_grid3 = joblib.load(savedir_models / 'scaler_grid3.sav')

# %% ------------------------------- XGBoost ------------------------------- #
model_name = "XGBoost_reg"

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.4, learning_rate = 0.22,
                max_depth = 11, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train, y_train)

filename = 'XGBoost_reg.sav'
joblib.dump(xg_reg, filename)

# %% ------------------------------- XGBoost feature importance ------------------------------- #
xgb.plot_importance(xg_reg, max_num_features=25)
plt.show()

# Fit model using each importance as a threshold
thresholds = -np.sort(-xg_reg.feature_importances_)
scores = []
n = []

for i in range(0, len(thresholds), 1):
    # select features using threshold
    selection = SelectFromModel(xg_reg, threshold=thresholds[i], prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.4, learning_rate = 0.22,
                max_depth = 11, alpha = 10, n_estimators = 20, verbosity = 0)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)

    y_pred = scaler_grid2.inverse_transform(pd.DataFrame(y_pred).values.reshape(-1, 1))

    accuracy = mean_absolute_percentage_error(y_pred, y_test)
    scores.append(accuracy)
    n.append(i)
    print("Thresh=%.5f, n=%d, MAPE: %.5f" % (thresholds[i], select_X_train.shape[1], accuracy))

# Plots the MAPE score for the different number of features
plt.scatter(n, scores)
plt.xlabel('Number of features')
plt.ylabel('Error')
plt.show()

# use the best threshold
# found by looking at the scatterplot above and the printed table
best_thres = 0.00084
selection = SelectFromModel(xg_reg, threshold=best_thres, prefit=True)
feature_idx = selection.get_support()
feature_idx
X_train_xgb_fs = X_train[X_train.columns[feature_idx]]
X_test_xgb_fs = X_test[X_train.columns[feature_idx]]

# creates a model with the best features
model_name = "XGBoost after feature selection"
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree= 0.97, gamma= 0.23, learning_rate= 0.05, max_depth= 24, min_child_weight= 7.0, n_estimators= 190, subsample= 0.31, verbosity = 0)
xg_reg.fit(X_train, y_train)

filename = 'XGBoost_reg_fs.sav'
joblib.dump(xg_reg, filename)
#xg_reg = joblib.load(filename)
y_pred = xg_reg.predict(X_test)

# scaling back to original scaling
y_pred = scaler_grid2.inverse_transform(pd.DataFrame(y_pred).values.reshape(-1, 1))
# train on the training data, and predicts all values in the test data
evaluate(y_observed, y_true, y_pred)

# %% ------------------------------- XGBoost hyperparameter optimization ------------------------------- #
#Bayesian optimazation
#This optimization runs for a while
#https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0

# Keep track of results
bayes_trials = Trials()

def objective(space):
    # Keep track of evals
    global ITERATION
    ITERATION += 1

    xg_reg = xgb.XGBRegressor(orobjective ='reg:squarederror',
                            n_estimators = space['n_estimators'],
                            max_depth = int(space['max_depth']),
                            learning_rate = space['learning_rate'],
                            gamma = space['gamma'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree'],
                            verbosity = 0
                            )

    time_split = TimeSeriesSplit(n_splits=10)
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = xg_reg, X = X_train_xgb_fs, y = y_train, cv = time_split, scoring ='neg_mean_squared_error' )
    CrossValMean = accuracies.mean()

    print("CrossValMean:", -CrossValMean)

    return{'loss':-CrossValMean, 'params': space, 'status': STATUS_OK }

# define the space that we want to search
# since we do not have any prior knowledge they are all uniform or choice
space = {
    'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}

# Global variable
global  ITERATION

ITERATION = 0

# tpe.suggest generates the agorithm for finding the next point to check
# fmin returns the best parameters
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 80, trials = bayes_trials, rstate = np.random.RandomState(50))
best_params = best
print(best_params)

# Sort the trials with lowest loss first
bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
print(bayes_trials_results[:2])
max_depth = []
learning_rate = []
n_estimators = []
gammas = []
min_child_weight = []
subsample = []
colsample_bytree = []
for i in range(80):
    m_d = bayes_trials_results[i]['params']['max_depth']
    max_depth.append(m_d)
    l_r = bayes_trials_results[i]['params']['learning_rate']
    learning_rate.append(l_r)
    n_e = bayes_trials_results[i]['params']['n_estimators']
    n_estimators.append(n_e)
    gamma = bayes_trials_results[i]['params']['gamma']
    gammas.append(gamma)
    m_c_w = bayes_trials_results[i]['params']['min_child_weight']
    min_child_weight.append(m_c_w)
    s = bayes_trials_results[i]['params']['subsample']
    subsample.append(s)
    c_b = bayes_trials_results[i]['params']['colsample_bytree']
    colsample_bytree.append(c_b)


bayes_params = pd.DataFrame({'max_depth': max_depth, 'learning_rate': learning_rate,
                'n_estimators': n_estimators, 'gamma': gammas, 'min_child_weight': min_child_weight,
                'subsample': subsample, 'colsample_bytree': colsample_bytree})

# Code for vizualizing the bayesian optimization
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18

# Density plots of the parameter distributions
for col in bayes_params:
    sns.kdeplot(bayes_params[col], label = 'Bayes Optimization', linewidth = 2)
    plt.legend()
    plt.xlabel(str(col))
    plt.ylabel('Density'); plt.title(str(col) + ' Distribution')
    plt.show()

model_name = "XGBoost_reg_best"

xg_reg = xgb.XGBRegressor(orobjective ='reg:squarederror',
                            n_estimators = best_params['n_estimators'],
                            max_depth = int(best_params['max_depth']),
                            learning_rate = best_params['learning_rate'],
                            gamma = best_params['gamma'],
                            min_child_weight = best_params['min_child_weight'],
                            subsample = best_params['subsample'],
                            colsample_bytree = best_params['colsample_bytree']
                            )

xg_reg.fit(X_train_xgb_fs, y_train)

filename = 'XGBoost_reg_best.sav'
joblib.dump(xg_reg, filename)
#xg_reg = joblib.load(filename)

# %% ------------------------------- Forecast ------------------------------- #
# we will forecast 6 days ahead, and fit the model on all available data up to this point
frames = [train.tail(lookback_window+nobs), test]
df_test = pd.concat(frames).drop(['grid1-load', 'grid1-loss', 'grid1-temp', 'grid3-load', 'grid3-loss', 'grid3-temp'], axis=1)

pred = []
y_test = df_test['grid2-loss'].shift(nobs).dropna()

for i in tqdm(range(test.shape[0])):
    # fit new model on the sliding window
    # the model does not handle conastant values so we can not use binary variable
    # if they are constant in the forecast window
    # important to have the sesonal time series, but check that they are not constant
    forecast_df = df_test.iloc[i:lookback_window+i]
    forecast_input = np.array(forecast_df)
    y_forecast = np.array(y_test.iloc[i:lookback_window+i])
    """
    xg_reg = xgb.XGBRegressor(orobjective ='reg:squarederror',
                                n_estimators = best_params['n_estimators'],
                                max_depth = int(best_params['max_depth']),
                                learning_rate = best_params['learning_rate'],
                                gamma = best_params['gamma'],
                                min_child_weight = best_params['min_child_weight'],
                                subsample = best_params['subsample'],
                                colsample_bytree = best_params['colsample_bytree']
                                )
    """
    # trains one model each day
    if (i % 24 == 0):
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree= 0.97, gamma= 0.23, learning_rate= 0.05, max_depth= 24, min_child_weight= 7.0, n_estimators= 30, subsample= 0.31, verbosity = 0)
        xg_reg.fit(forecast_input, y_forecast)

    pred_value = xg_reg.predict([X_test.iloc[i, :].values])
    pred.append(pred_value)

#print(pred)
# %% ------------------------------- Evaluate ------------------------------- #

y_pred = scaler_grid2.inverse_transform(np.array(pred).reshape(-1, 1))
mae, rmse, mape = evaluate(y_observed, y_true, y_pred)
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
