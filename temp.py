#league analytics:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("E:\\ProjectDataFolder\\LOL")

raw_data = pd.read_csv("high_diamond_ranked_10min.csv")

raw_data.columns

# we remove game id, red first blood, red gold diff and red exp diff. these are
# either useless or direct counterparts of another, e.g. red deaths == 
# blue kills
wdata = raw_data[['blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood',
                  'blueKills', 'blueDeaths', 'blueAssists','blueEliteMonsters',
                  'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
                  'blueAvgLevel', 'blueTotalExperience',
                  'blueTotalJungleMinionsKilled', 'blueGoldDiff',
                  'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin',
                  'redWardsPlaced', 'redWardsDestroyed', 'redAssists',
                  'redEliteMonsters', 'redDragons', 'redHeralds',
                  'redTowersDestroyed','redAvgLevel', 'redTotalExperience', 
                  'redTotalJungleMinionsKilled','redCSPerMin', 'redGoldPerMin',
                  'blueWins']]


wdata.dtypes

#corr matrix:
plt.figure(figsize=(20,10))
sns.heatmap(wdata.corr(), annot=True)
plt.show()

#TODO: feature engineering, a lot of data is "codependent"
#TODO: usual  classifier + feature importance, maybe shap now. 

