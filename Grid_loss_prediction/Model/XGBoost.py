'''
Contents:

- Imports
- General settings
- Read data
- Feature importance
- Hyperparameter optimization
- Forecasting
- Evaluation

'''

import numpy as np
import pandas as pd
import seaborn as sns
import joblib as joblib
import matplotlib.pyplot as plt
from pathlib import Path
from evaluation import evaluate, mean_absolute_percentage_error
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
import xgboost as xgb
from hyperopt import hp, tpe, fmin
from hyperopt import Trials
from hyperopt import STATUS_OK
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit


def main():

    ''' General settings '''

    savedir_models = Path('Models/').resolve()
    loaddir = Path('Data/').resolve()
    nobs = 144 # 6 days
    lookback_window = 24*90 # 90 days
    grid_number = 1 # select which grid to predict : 0 -> grid1, 1 -> grid2, 2 -> grid3
    columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']
    target_col = columns_to_predict[grid_number] # select which column to predict
    MAPE = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


    ''' Read data '''

    # x-values
    pickle_path = Path('Data/serialized/x_train_with_lag').resolve()
    X_train = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/x_test_with_lag').resolve()
    X_test = pd.read_pickle(pickle_path)

    #y-values
    pickle_path = Path('Data/serialized/y_train_grid1').resolve()
    y_train_grid1 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_test_grid1').resolve()
    y_test_grid1 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_train_grid2').resolve()
    y_train_grid2 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_test_grid2').resolve()
    y_test_grid2 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_train_grid3').resolve()
    y_train_grid3 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_test_grid3').resolve()
    y_test_grid3 = pd.read_pickle(pickle_path)

    # creates dataframes for evaluation
    observed = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
    test_true = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)
    y_true = test_true[columns_to_predict]
    y_observed = observed[columns_to_predict][10000:]

    # scalers for inverse transform after predicting
    scaler_grid1 = joblib.load(savedir_models / 'scaler_grid1.sav')
    scaler_grid2 = joblib.load(savedir_models / 'scaler_grid2.sav')
    scaler_grid3 = joblib.load(savedir_models / 'scaler_grid3.sav')

    y_train = [y_train_grid1, y_train_grid2, y_train_grid3][grid_number] # select the y_train for the selected grid
    y_test = [y_test_grid1, y_test_grid2, y_test_grid3][grid_number]
    scaler = [scaler_grid1, scaler_grid2, scaler_grid3][grid_number] # select the scaler for the selected grid


    ''' Feature importance '''

    print("\nFeature importance...")

    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree= 0.97, gamma= 0.23, learning_rate= 0.05, max_depth= 24, min_child_weight= 7.0, n_estimators= 190, subsample= 0.31, verbosity = 0)
    xg_reg.fit(X_train, y_train)

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
        selection_model.fit(select_X_train, y_train_grid1)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)

        y_pred = scaler.inverse_transform(pd.DataFrame(y_pred).values.reshape(-1, 1))
        y_test_selected = scaler.inverse_transform(pd.DataFrame(y_test_grid1).values.reshape(-1, 1))
        accuracy = mean_absolute_percentage_error(y_pred, np.array(y_test_selected))
        scores.append(accuracy)
        n.append(i)
        print("Thresh=%.5f, n=%d, MAPE: %.5f" % (thresholds[i], select_X_train.shape[1], accuracy))

    # Plots the MAPE score for the different number of features
    plt.scatter(n, scores)
    plt.xlabel('Number of features')
    plt.ylabel('MAPE')
    plt.show()

    # use the best threshold
    # found by looking at the scatterplot above and the printed table
    best_thres = 0.00084
    selection = SelectFromModel(xg_reg, threshold=best_thres, prefit=True)
    feature_idx = selection.get_support()
    X_train_fs = X_train[X_train.columns[feature_idx]]
    X_test_fs = X_test[X_train.columns[feature_idx]]

    # creates a model with the best features
    model_name = "XGBoost after feature selection"
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree= 0.97, gamma= 0.23, learning_rate= 0.05, max_depth= 24, min_child_weight= 7.0, n_estimators= 190, subsample= 0.31, verbosity = 0)
    xg_reg.fit(X_train_fs, y_train)
    y_pred = xg_reg.predict(X_test_fs)
    # scaling back to original scaling
    y_pred = scaler.inverse_transform(pd.DataFrame(y_pred).values.reshape(-1, 1))
    # train on the training data, and predicts all values in the test data
    mae, rmse, mape = evaluate(np.array(y_observed[target_col]), np.array(y_true[target_col]), y_pred)
    print("The whole test set predicted from the best features.")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)


    ''' Hyperparameter optimization '''

    print("\nHyperparameter optimization")
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
        accuracies = cross_val_score(estimator = xg_reg, X = X_train_fs, y = y_train, cv = time_split, scoring = MAPE ) # important to use MAPE here since some of the grids change dramatically over time
        CrossValMean = accuracies.mean()

        print("CrossVal mean of MAPE:", -CrossValMean)

        return{'loss': -CrossValMean, 'params': space, 'status': STATUS_OK }

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


    ''' Forecast '''

    print("\nForecast...")
    # we will forecast 6 days ahead, and fit the model on all available data up to this point
    x_forecast = pd.concat([X_train.tail(lookback_window+nobs), X_test])

    y_forecast = pd.concat([y_train.tail(lookback_window+nobs), y_test])

    pred = []

    for i in tqdm(range(X_test.shape[0])):
        # fit new model on the sliding window
        # the model does not handle conastant values so we can not use binary variable
        # if they are constant in the forecast window
        # important to have the sesonal time series, but check that they are not constant
        x_forecast_slice = np.array(x_forecast.iloc[i:lookback_window+i])
        y_forecast_slice = np.array(y_forecast.iloc[i:lookback_window+i])

        if (i%24 == 0):
            xg_reg = xgb.XGBRegressor(orobjective ='reg:squarederror',
                                        n_estimators = best_params['n_estimators'],
                                        max_depth = int(best_params['max_depth']),
                                        learning_rate = best_params['learning_rate'],
                                        gamma = best_params['gamma'],
                                        min_child_weight = best_params['min_child_weight'],
                                        subsample = best_params['subsample'],
                                        colsample_bytree = best_params['colsample_bytree']
                                        )
            # trains one model each day
            xg_reg.fit(x_forecast_slice, y_forecast_slice)

        pred_value = xg_reg.predict([X_test.iloc[i, :].values])
        pred.append(pred_value)


    ''' Evaluate '''

    y_pred = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
    mae, rmse, mape = evaluate(np.array(y_observed[target_col]), np.array(y_true[target_col]), np.array(y_pred))

    print("\nEvaluate...")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)


if __name__ == "__main__":
    main()
