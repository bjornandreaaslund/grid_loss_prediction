'''
Contents:

- Imports
- General settings
- Read data
- Preparation of data
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
from evaluation_metric import evaluate, mean_absolute_percentage_error
from nbeats_keras.model import NBeatsNet
from sklearn.metrics import make_scorer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from hyperopt import hp, tpe, fmin
from hyperopt import Trials
from hyperopt import STATUS_OK
import tensorflow as tf
from tqdm import tqdm


def prepare_data(time_series, forecast_length, backcast_length, point = True):
    '''
    Transforms timeseries of length n to (n//backcast_length, backcast_length, 1)

    '''

    x = []
    y = []
    examples = len(time_series)-(backcast_length+forecast_length)
    if point == False:
        for i in range(examples):
            x.append(time_series[i:i+backcast_length])
            y.append(time_series[i+backcast_length:i+backcast_length+forecast_length])
        x = np.array(x).reshape(examples,backcast_length, 1)
        y = np.array(y).reshape(examples, forecast_length, 1)
    if point == True:
        for i in range(examples):
            x.append(time_series[i:i+backcast_length])
            y.append([i+backcast_length+forecast_length])
        x = np.array(x).reshape(examples,backcast_length, 1)
        y = np.array(y).reshape(examples, 1, 1)
    return x, y


def build_objective(timeseries_data):

    def objective(space):
        '''
        TODO: add docstring

        '''

        # Keep track of evals
        global ITERATION
        ITERATION += 1
        x, y = prepare_data(time_series=timeseries_data, forecast_length=FORECAST_LENGTH, backcast_length=space['backcast_length'], point = True)
        print("ok")
        nbeats = NBeatsNet(input_dim=INPUT_DIM, backcast_length=space['backcast_length'], forecast_length=OUTPUT_DIM,
                        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=space['nb_blocks_per_stack'],
                        thetas_dim=(4, 4), share_weights_in_stack=space['share_weights_in_stack'], hidden_layer_units=space['hidden_layer_units'])
        print("2")
        nbeats.compile_model(loss='mae', learning_rate=space['learning_rate'])
        # Split data into training and testing datasets.
        c = x.shape[0] // 10
        x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        # Train the model.
        nbeats.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)
        loss = nbeats.evaluate(x_test, y_test)

        print("CrossVal mean of MAE:", loss)

        return{'loss': loss, 'params': space, 'status': STATUS_OK }
    
    return objective


def main():

    ''' Read data '''

    # x-values
    pickle_path = Path('Data/serialized/x_train').resolve()
    X_train = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/x_test').resolve()
    X_test = pd.read_pickle(pickle_path)

    # exogenous variables
    exo_train = train[['grid2-load', 'gird2-temp']]
    exo_ttest = test[['grid2-load', 'gird2-temp']]

    timeseries = pd.concat([X_train, X_test])[target_col]
    timeseries_train = timeseries.head(-X_test.shape[0])
    timeseries_test = timeseries.tail(X_test.shape[0])

    # creates dataframes for evaluation
    observed = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
    test_true = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)
    y_true = test_true[columns_to_predict]
    y_observed = observed[columns_to_predict][10000:]

    # scalers for inverse transform after predicting
    scaler_grid1 = joblib.load(savedir_models / 'scaler_grid1.sav')
    scaler_grid2 = joblib.load(savedir_models / 'scaler_grid2.sav')
    scaler_grid3 = joblib.load(savedir_models / 'scaler_grid3.sav')

    scaler = [scaler_grid1, scaler_grid2, scaler_grid3][GRID_NUMBER] # select the scaler for the selected grid


    ''' Hyperparameter optimization '''

    print("\nHyperparameter optimization")

    #Bayesian optimazation
    #This optimization runs for a while
    #https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0

    bayes_trials = Trials()     # Keep track of results

    # define the space that we want to search
    # since we do not have any prior knowledge they are all uniform or choice
    space = {
        'backcast_length': hp.choice('backcast_length', range(NOBS, 7*NOBS, NOBS)),
        'nb_blocks_per_stack': hp.choice('nb_blocks_per_stack', range(1, 4, 1)),
        'share_weights_in_stack': hp.choice('share_weights_in_stack', [True, False]),
        'hidden_layer_units': hp.choice('hidden_layer_units', [32, 64, 128]),
        'learning_rate': hp.choice('learning_rate', [1e-6, 1e-5, 1e-4, 1e-3])}

    # tpe.suggest generates the agorithm for finding the next point to check
    # fmin returns the best parameters
    evals = 40
    objective = build_objective(timeseries_train)
    best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = evals, trials = bayes_trials, rstate = np.random.RandomState(50))
    best_params = best
    print(best_params)

    # Sort the trials with lowest loss first
    bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
    print(bayes_trials_results[:2])
    backcast_length = []
    nb_blocks_per_stack = []
    share_weights_in_stack = []
    hidden_layer_units = []
    learning_rate = []
    for i in range(evals):
        b_l = bayes_trials_results[i]['params']['backcast_length']
        backcast_length.append(b_l)
        b_per_s = bayes_trials_results[i]['params']['nb_blocks_per_stack']
        nb_blocks_per_stack.append(b_per_s)
        s_w = bayes_trials_results[i]['params']['share_weights_in_stack']
        share_weights_in_stack.append(s_w)
        h_l = bayes_trials_results[i]['params']['hidden_layer_units']
        hidden_layer_units.append(h_l)
        l_r = bayes_trials_results[i]['params']['learning_rate']
        learning_rate.append(l_r)


    bayes_params = pd.DataFrame({'backcast_length': backcast_length, 'nb_blocks_per_stack': nb_blocks_per_stack,
                'share_weights_in_stack': share_weights_in_stack, 'hidden_layer_units': hidden_layer_units,
                'learning_rate': learning_rate})

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

    model_name = "NBeats_best"

    nbeats = NBeatsNet(input_dim=INPUT_DIM, backcast_length=best_params['backcast_length'], forecast_length=OUTPUT_DIM,
                    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=best_params['nb_blocks_per_stack'],
                    thetas_dim=(4, 4), share_weights_in_stack=space['share_weights_in_stack'], hidden_layer_units=best_params['hidden_layer_units'])

    nbeats.compile_model(loss='mae', learning_rate=best_params['learning_rate'])

    nbeats.save(savedir_models.resolve() / 'n_beats_grid1.h5')


    ''' Forecast '''

    print("\nForecast...")
    # we will forecast 6 days ahead, and fit the model on all available data up to this point

    forecast_data = timeseries.tail(X_test.shape[0]+ NOBS + LOOKBACK_WINDOW)
    pred = []
    horizons = np.arange(1, 8)
    backcast_length=24
    ensemble = pd.Dataframe()

    #nbeats = NBeatsNet.load(savedir_models / 'n_beats_model.h5')
    nbeats = NBeatsNet(input_dim=INPUT_DIM, backcast_length=backcast_length,
                    forecast_length=OUTPUT_DIM,
                    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=1,
                    thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=128)
    nbeats.compile_model(loss='mae', learning_rate=1e-5)

    for h in horizons: # creates predictions for each of the horizons to make an ensemble
        backcast_length = 24*h
        for i in tqdm(range(X_test.shape[0])):
            # fit new model on the sliding window
            # the model does not handle conastant values so we can not use binary variable
            # if they are constant in the forecast window
            # important to have the sesonal time series, but check that they are not constant
            forecast_slice = np.array(timeseries.iloc[i:LOOKBACK_WINDOW+i])

            x, y = prepare_data(forecast_slice, FORECAST_LENGTH, backcast_length, point=True)

            if (i%24 == 0):

                # trains one model each day
                nbeats.fit(x, y, epochs=10, batch_size=128)

            pred_value = nbeats.predict(np.array(timeseries.iloc[LOOKBACK_WINDOW+i+1-backcast_length:LOOKBACK_WINDOW+i+1]).reshape(1, backcast_length, 1))
            pred.append(pred_value[:,0, 0])
        ensemble[str(h)] = pred
        pred = []


    ''' Evaluate '''

    # selects the median of the predictions to handle outliers
    median = ensemble.median(axis=0) # axis=0 to find the median of each row

    y_pred = scaler.inverse_transform(np.array(median).reshape(-1, 1))
    mae, rmse, mape = evaluate(np.array(y_observed[target_col]), np.array(y_true[target_col]), np.array(y_pred))
    
    print("\nEvaluate...")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)


if __name__ == "__main__":

    ''' General settings '''
    
    GRID_NUMBER = 1 # select which grid to predict : 0 -> grid1, 1 -> grid2, 2 -> grid3
    NOBS = 144 # 6 days
    LOOKBACK_WINDOW = 24*90 # 90 days
    FORECAST_LENGTH = 6*24 # 6 days
    INPUT_DIM = 1
    OUTPUT_DIM = 1
    MAPE = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    global  ITERATION
    ITERATION = 0

    savedir_models = Path('Models/')
    loaddir = Path('Data/')

    columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']
    target_col = columns_to_predict[GRID_NUMBER] # select which column to predict

    if OUTPUT_DIM ==1:
        point = True
    else:
        point = False


    ''' Execute NBEATS forcasting '''

    main()
