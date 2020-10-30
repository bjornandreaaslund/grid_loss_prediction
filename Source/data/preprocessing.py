import os
import gc
import datetime as datetime
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import joblib as joblib
from itertools import groupby, count
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler

# General Settings
pd.options.mode.chained_assignment = None
loaddir = Path('Data/').resolve()
savedir_log = Path('Log/').resolve()
savedir_models = Path('Models/').resolve()
savedir_meta = Path('Data/meta/').resolve()
if not os.path.exists(savedir_log):
    os.mkdir(os.getcwd() + savedir_log)
if not os.path.exists(savedir_meta):
    os.mkdir(os.path.join(os.getcwd(), savedir_meta))

df_columns = {  'Grid_data' : ['Unnamed: 0', 'demand', 'grid1-load', 'grid1-loss','grid1-temp', 'grid2-load', 'grid2-loss','grid2_1-temp',
                                'grid2_2-temp', 'grid3-load', 'grid3-loss', 'grid3-temp', 'has incorrect data'],
                'Prophet' : ['grid1-loss-prophet-daily', 'grid1-loss-prophet-pred', 'grid1-loss-prophet-trend', 'grid1-loss-prophet-weekly',
                            'grid1-loss-prophet-yearly', 'grid2-loss-prophet-daily', 'grid2-loss-prophet-pred','grid2-loss-prophet-trend',
                            'grid2-loss-prophet-weekly','grid2-loss-prophet-yearly', 'grid3-loss-prophet-pred', 'grid3-loss-prophet-trend',
                            'grid3-loss-prophet-weekly', 'grid3-loss-prophet-yearly'],
                'Seasonal' : ['season_x', 'season_y', 'month_x', 'month_y', 'week_x', 'week_y',
                            'weekday_x', 'weekday_y', 'holiday', 'hour_x', 'hour_y'],
                'Temperature' : ['grid1-temp', 'grid2_1-temp', 'grid2_2-temp', 'grid3-temp'],
                'Timestamp' : ['Unnamed: 0']
            }

# grid 3 has no values before this index and we do not want to impute all these values
start_index_grid3 = 6622

# grid 2 loss has wrong measurements between these indexes, and that value will not be used
sensor_error_start = 1079
sensor_error_end = 2591

train = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)

# %% ---------------------- Grouping data and saving the in Data/processed --------------------- #
grid_data = train[df_columns['Grid_data']]
grid_data.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
grid_data.to_csv(loaddir.joinpath('processed/grid_data.csv'))

prophet = train[df_columns['Prophet']]
prophet.to_csv(loaddir.joinpath('processed/prophet_data.csv'))

seasonal = train[df_columns['Seasonal']]
seasonal.to_csv(loaddir.joinpath('processed/seasonal_data.csv'))

del prophet, seasonal

# %% ---------------------- Data cleaning -------------------------------------#
print("\nData cleaning...")
C_to_K = 273

# check that the temerature is between 223 and 323
for temp in df_columns['Temperature']:
    print("Number of measurments with temerature above 323 K for " + temp + " :" + str(train[train[temp] > 50+C_to_K].shape[0]))
    print("Number of measurments with temerature below 223 K for " + temp + " :" + str(train[train[temp] < C_to_K-50].shape[0])+ "\n")

# check that all the timestamp has an interval of one hour
check = True
for i in range(train.shape[0]-1):
    datetime2 = datetime.datetime.strptime(train.iloc[i+1, 0], '%Y-%m-%d %H:%M:%S')
    datetime1 = datetime.datetime.strptime(train.iloc[i, 0], '%Y-%m-%d %H:%M:%S')
    one_hour = datetime.datetime.strptime('1:00:00', '%H:%M:%S')
    if (datetime2 - datetime1 != datetime.timedelta(hours=1)):
        check = False
result = 'All timestamps have an intervall of one hour' if check else 'All timestamps do not have an intervall of one hour\n'
print(result)

# assume that there is not possible to have 0 loss if the load is positive.

zero_loss = train.loc[(train['grid3-loss'] == 0) & (train['grid3-load'] > 0)].index
zero_loss = zero_loss.to_list()

train['grid3-loss'].replace(0.0,np.NaN, inplace=True)

# remove wrong measurement from grid2-loss
train['grid2-loss'][sensor_error_start:sensor_error_end].replace(train['grid2-loss'][train['grid2-loss'] > 0], np.nan, inplace=True)

# %% ------------------------------- Imputation -------------------------------- #
print("\nImputing missing values...")

# columns with nan, which need imputation
nancols = list(grid_data.columns[grid_data.isna().sum() != 0])
nancols0 = ['demand', 'grid1-loss', 'grid1-temp', 'grid2-loss', 'grid2_1-temp', 'grid2_2-temp', 'grid3-load', 'grid3-loss', 'grid3-temp']
nancols_strp = [col for col in nancols0 for nancol in nancols if nancol.startswith(col)]
nancols_bin = {col:train[col].dropna().isin([0,1]).all() for col in nancols}

# threshold for correlation value for a column to be used in imputation
t = 0.6

# Create dictionary corr with for each nancol corresponding features with correlation higher than threshold t
corr = {}
for nancol in nancols:
    corr.update({nancol : pd.DataFrame(columns=['c'])})
    for col in df_columns['Grid_data'][1:-1] + df_columns['Seasonal']:
        corr[nancol].loc[col, 'c'] = np.abs(train[nancol].corr(train[col]))
    corrcol = list(corr[nancol][corr[nancol].c > t].index)
    if len(corrcol) > 1:
        print(list(set(corrcol) - set(nancol)))
        print('\n')
        corrcol = [nancol] + list(set(corrcol) - {nancol}) #bring nancol to front
        corr.update({nancol:corrcol})
    else:
        print("Feature with not enough correlations:", nancol)
        alt_corrcol = list(corr[nancol].c.sort_values()[-3:].index)
        alt_corrcol = [nancol] + list(set(alt_corrcol) - {nancol}) #bring nancol to front
        corr.update({nancol:alt_corrcol})

print("The following columns are correlated with the nancols", corr)

# Features with only correlations with itself
dropnancols = list(set(nancols) - set([nancol for nancol_strp, nancol, corrcols in zip(nancols_strp, corr.keys(), corr.values()) for col in corrcols if col.startswith(nancol_strp) == False]))
print("\nFeatures with only correlations with itself:", dropnancols)
for col in dropnancols:
    del corr[col]
gc.collect()

imputed_rows = []

# imputation
for col in nancols:
    if (col in corr) and (col == "grid3-loss" or col == "grid3-load"):
        print("Imputation will be done with ExtraTreesRegressor for: " + col)
        imputed_rows += train[col][start_index_grid3:].index[train[col][start_index_grid3:].apply(np.isnan)].to_list()
        train[col][start_index_grid3:] = pd.Series(IterativeImputer(ExtraTreesRegressor(n_estimators=30, min_samples_split=0.05, n_jobs = -1)).fit_transform(train[corr[col]].iloc[start_index_grid3:,] )[:,0]).copy(deep=True)
    elif col in corr:
        print("Imputation will be done with ExtraTreesRegressor for: " + col)
        imputed_rows += train[col].index[train[col].apply(np.isnan)].to_list()
        train[col] = pd.DataFrame(IterativeImputer(ExtraTreesRegressor(n_estimators=30, min_samples_split=0.05, n_jobs = -1)).fit_transform(train[corr[col]])[:,0]).copy(deep=True)

# visualize which rows which is labeled incorrect, but were not cleaned
imputed_rows = list(set(imputed_rows))
incorrect_index = np.array(train.index[train['has incorrect data'] == True].tolist())
not_fixed = np.setdiff1d(incorrect_index, imputed_rows)

def as_range(iterable): # not sure how to do this part elegantly
    l = list(iterable)
    if len(l) > 1:
        return '{0}-{1}'.format(l[0], l[-1])
    else:
        return '{0}'.format(l[0])
print('\nImputed rows')
print(','.join(as_range(g) for _, g in groupby(imputed_rows, key=lambda n, c=count(): n-next(c))))
print('\nRows with incorrect data')
print(','.join(as_range(g) for _, g in groupby(incorrect_index, key=lambda n, c=count(): n-next(c))))

# print out the ranges of incorrect data
index_ranges = ','.join(as_range(g) for _, g in groupby(not_fixed, key=lambda n, c=count(): n-next(c)))
print("\nThe incorrect data which is not handeled:")
for index_range in index_ranges.split(','):
    print(index_range)

# data which will be used in the training
x_train = train[df_columns['Grid_data'][1:-1] + df_columns['Seasonal']]

pickle_path = Path('Data/serialized/processed_x_train_pickle')
x_train.to_pickle(pickle_path)

# %% ------------------------------- Scaling -------------------------------- #
print('\nScaling and desesonalizing...')
# we use robust scaler since we have some anomalies
scaler = RobustScaler().fit(x_train)
scaler_filename = "scaler.save"
joblib.dump(scaler, savedir_models / scaler_filename)
scaler = joblib.load(savedir_models / scaler_filename)
x_train[x_train.columns] = scaler.transform(x_train)

# %% ------------------------------- Serialize -------------------------------- #
print('\nSaving preprocessed data...')
pickle_path = Path('Data/serialized/processed_x_train_scaled_pickle')
x_train.to_pickle(pickle_path)

print('Preprocessing done!')
