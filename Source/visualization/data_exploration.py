import os
import gc
import datetime as datetime
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import groupby, count

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

# see the ranges of incorrect data
print("There are ", train['has incorrect data'].value_counts().iloc[1], " rows with incorrect data\n")
index_list = np.array(train.index[train['has incorrect data'] == True].tolist())

def as_range(iterable): # not sure how to do this part elegantly
    l = list(iterable)
    if len(l) > 1:
        return '{0}-{1}'.format(l[0], l[-1])
    else:
        return '{0}'.format(l[0])

# print out the ranges of incorrect data
index_ranges = ','.join(as_range(g) for _, g in groupby(index_list, key=lambda n, c=count(): n-next(c)))
print("The incorrect data is located in these ranges:")
for index_range in index_ranges.split(','):
    print(index_range)
print('\n')


# %% ---------------------- Data cleaning -------------------------------------#
C_to_K = 273

# check that the temerature is between 223 and 323
for temp in df_columns['Temperature']:
    print("Number of measurments with temerature above 323 K for " + temp + " :" + str(train[train[temp] > 40+C_to_K].shape[0]))
    print("Number of measurments with temerature below 223 K for " + temp + " :" + str(train[train[temp] < C_to_K-40].shape[0])+ "\n")

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
train = train.set_index('Unnamed: 0')

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

def plot_wrong_sensordata():
    plt.plot(grid_data['grid2-loss'])
    plt.title('Wrong measurement for grid2 loss')
    plt.xlim(sensor_error_start-100, sensor_error_end+100)
    plt.savefig(savedir_log+'wrong_measurement_grid2_loss')
    plt.show()

def plot_temperature():
    plt.plot(train[df_columns['Temperature']])
    plt.title('The temperature for each of the grids.')
    plt.xlabel('Time')
    plt.ylabel('Temperature [K]')
    plt.savefig(savedir_log+'temperature')
    plt.show()

def plot_loss():
    plt.plot(train[['grid1-loss', 'grid2-loss', 'grid2-loss']])
    plt.title('The loss from each grid.')
    plt.xlabel('Time')
    plt.ylabel('Energy [MWh]')
    plt.savefig(savedir_log+'loss')
    plt.show()

def plot_load():
    plt.plot(train[['grid1-load', 'grid2-load', 'grid2-load']])
    plt.title('The load from each grid.')
    plt.xlabel('Time')
    plt.ylabel('Energy [MWh]')
    plt.savefig(savedir_log+'load')
    plt.show()
