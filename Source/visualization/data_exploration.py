import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


train = pd.read_csv("../../Data/raw/train.csv", header=0)
print(train.columns)

df = train[['demand', 'grid1-load', 'grid1-loss', 'grid1-temp', 'grid2-load', 'grid2-loss', 'grid2_1-temp', 'grid2_2-temp', 'grid3-load', 'grid3-loss', 'grid3-temp']]

print(df.describe())

print(df.isnull().sum())

sns.clustermap(df.corr())
plt.show()
print(len(np.fft.hfft(df['grid1-load'].to_numpy())))
print(len(df['demand']))

df.plot()
plt.show()
