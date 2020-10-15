import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# General Settings
loaddir = "../../Data/"
savedir_log = "../../Log/"
if not os.path.exists(savedir_log):
    os.mkdir(savedir_meta)
savedir_meta = "../../Data/meta/"
if not os.path.exists(savedir_meta):
    os.mkdir(savedir_meta)


train = pd.read_csv(loaddir + "raw/train.csv", header=0)

# %% ---------------------- Removing prophet values and _y values --------------------- #
df = train[['demand', 'grid1-load', 'grid1-loss', 'grid1-temp', 'grid2-load', 'grid2-loss', 'grid2_1-temp', 'grid2_2-temp', 'grid3-load', 'grid3-loss', 'grid3-temp', 'season_x', 'month_x', 'week_x',
       'weekday_x', 'holiday', 'hour_x',
       'has incorrect data']]


# %% ---------------------- Descriptives before cleaning --------------------- #
# Number of rows and columns
print("Rows:              ", df.shape[0])
print("Columns:           ", df.shape[1])

# Overview of numerical columns
df.describe()
df.describe().T.to_latex(savedir_meta + 'descriptives_raw.tex', float_format='%.2f')
pd.DataFrame(df.shape[1]*[''], index=df.columns, columns=['Description']).to_latex(savedir_meta + 'features_raw.tex')

print("Rows with NaN demand")
print(df[df['demand'].isna()])

print("Number of nan values", df.isnull().sum())

# %% ---------------------- Data cleaning -------------------------------------#
C_to_K = 273

grid_temp = ['grid1-temp', 'grid2_1-temp', 'grid2_2-temp', 'grid3-temp']
for temp in grid_temp:
    print("Number of measurments with temerature above 323 K for " + temp + " :" + str(df[df[temp] > 50+C_to_K].shape[0]))
    print("Number of measurments with temerature below 223 K for " + temp + " :" + str(df[df[temp] < C_to_K-50].shape[0])+ "\n")

def evaluate_seasonaldata():
    # season has 4 unique values
    print(train['season_x'].unique())
    # has 12 unique values
    print(train['month_x'].unique())
    # has
    print(len(train['hour_x'].unique()))

    plt.plot(train['month_x'])
    plt.title("month")
    plt.savefig(savedir_log+"month_x")
    plt.show()
    plt.title("season")
    plt.plot(train['season_x'])
    plt.savefig(savedir_log+"season_x")
    plt.show()
    plt.title("weekday_x")
    plt.plot(train['weekday_x'])
    plt.savefig(savedir_log+"weekday_x")
    plt.show()
    plt.title("hour_x")
    plt.plot(train['hour_x'])
    plt.savefig(savedir_log+"hour_x")
    plt.show()
    plt.title("holiday")
    plt.plot(train['holiday'])
    plt.savefig(savedir_log+"holiday")
    plt.show()

def print_corr_matrix():
    sns.clustermap(df.corr())
    plt.title("Correlation matrix")
    plt.savefig(savedir_log+"corr_matrix")
    plt.show()
