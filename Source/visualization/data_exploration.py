import os
import gc
import datetime as datetime
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

# General Settings
loaddir = Path('Data/').resolve()
savedir_log = Path('Log/').resolve()
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

# %% ---------------------- Descriptives before cleaning --------------------- #
# Number of rows and columns
print("Rows:              ", train.shape[0])
print("Columns:           ", train.shape[1])

# Overview of numerical columns
# TODO: describe appropirate groups of columns
grid_data.describe()
grid_data.describe().T.to_latex(savedir_meta.joinpath('descriptives_raw.tex'), float_format='%.2f')
pd.DataFrame(grid_data.shape[1]*[''], index=grid_data.columns, columns=['Description']).to_latex(savedir_meta.joinpath('features_raw.tex'))

print("Rows with NaN demand:")
print(grid_data[grid_data['demand'].isna()], '\n')

print("Number of nan values:")
print(grid_data.isnull().sum(), '\n')

# %% ---------------------- Data cleaning -------------------------------------#
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
result = 'All timestamps have an intervall of one hour' if check else 'All timestamps do not have an intervall of one hour'
print(result)

# %% ---------------------- Visualize data -------------------------------------#

def evaluate_seasonaldata():
    for col in df_columns['Seasonal']:
        print(train[col].unique())
        print(col, ' has ', len(train[col].unique()), 'unique values.')
        plt.plot(train[col])
        plt.title(col)
        plt.savefig(savedir_log+col)
        plt.show()

def print_corr_matrix():
    sns.clustermap(train[df_columns['Grid_data']].corr())
    plt.title("Correlation matrix")
    plt.savefig(savedir_log+"corr_matrix")
    plt.show()

# %% ------------------------------- Imputation -------------------------------- #


# columns with nan, which need imputation
nancols = list(grid_data.columns[grid_data.isna().sum() != 0])
nancols0 = ['demand', 'grid1-loss', 'grid1-temp', 'grid2-loss', 'grid2_1-temp', 'grid2_2-temp', 'grid3-load', 'grid3-loss', 'grid3-temp']
nancols_strp = [col for col in nancols0 for nancol in nancols if nancol.startswith(col)]
nancols_bin = {col:train[col].dropna().isin([0,1]).all() for col in nancols}

t = 0.6

# Create dictionary corr with for each nancol corresponding features with correlation higher than threshold t
# TODO: determine wich columns to impute
corr = {}
for nancol in nancols:
    corr.update({nancol : pd.DataFrame(columns=['c'])})
    for col in df_columns['Grid_data'][1:-1] + df_columns['Seasonal']:
        corr[nancol].loc[col, 'c'] = np.abs(train[nancol].corr(train[col]))
    corrcol = list(corr[nancol][corr[nancol].c > t].index)
    if len(corrcol) > 1:
        print(list(set(corrcol) - set(nancol)))
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
print("Features with only correlations with itself:", dropnancols)
for col in dropnancols:
    del corr[col]
gc.collect()

# imputation
for col in nancols:
    #plt.figure()
    #df[col].hist()
    if col in corr:
        print("Imputation will be done with ExtraTreesRegressor for: " + col)
        train[col] = pd.DataFrame(np.round(IterativeImputer(ExtraTreesRegressor(n_estimators=30, min_samples_split=0.05, n_jobs = -1)).fit_transform(train[corr[col]])[:,0]))
    #else:
    #    print("Imputation will be done with most frequent value for: " + col)
    #    df[col] = pd.DataFrame(np.round(SimpleImputer(missing_values=np.nan,  strategy='most_frequent').fit_transform(df[[col]])[:,0]))



# %% ------------------------------- Serialize -------------------------------- #

pickle_path = Path('Data/serialized/processed_data_pickle')
train.to_pickle(pickle_path)
