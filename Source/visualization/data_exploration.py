import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# General Settings
loaddir = "../../Data/"
savedir = "../../Log/"
if not os.path.exists(savedir):
    os.mkdir(savedir)

train = pd.read_csv(loaddir + "raw/train.csv", header=0)
print(train.columns)

# season has 4 unique values
print(train['season_x'].unique())
# has 12 unique values
print(train['month_x'].unique())
# has
print(len(train['hour_x'].unique()))

plt.plot(train['month_x'])
plt.title("month")
plt.savefig(savedir+"month_x")
plt.show()
plt.title("season")
plt.plot(train['season_x'])
plt.savefig(savedir+"season_x")
plt.show()
plt.title("weekday_x")
plt.plot(train['weekday_x'])
plt.savefig(savedir+"weekday_x")
plt.show()
plt.title("hour_x")
plt.plot(train['hour_x'])
plt.savefig(savedir+"hour_x")
plt.show()
plt.title("holiday")
plt.plot(train['holiday'])
plt.savefig(savedir+"holiday")
plt.show()

# only select the columns which we need
df = train[['demand', 'grid1-load', 'grid1-loss', 'grid1-temp', 'grid2-load', 'grid2-loss', 'grid2_1-temp', 'grid2_2-temp', 'grid3-load', 'grid3-loss', 'grid3-temp']]

#
print("Rows with NaN demand")
print(df[df['demand'].isna()])

print(df.describe())

print(df.isnull().sum())

"""
sns.clustermap(df.corr())
plt.show()
print(len(np.fft.hfft(df['grid1-load'].to_numpy())))
print(len(df['demand']))

df.plot()
plt.show()
"""
