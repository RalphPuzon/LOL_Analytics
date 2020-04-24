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

# DONE: feature engineering, a lot of data is "codependent"

# check completeness:
    
print(wdata.isnull().sum()/len(wdata))

# POA: usual  classifier + feature importance, maybe shap now. 

#make MOF and target:
X = wdata.iloc[:, :-1]
y = wdata.iloc[:,-1].to_frame()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train,
                             cv = 5)
accmean = accuracies.mean()
stddev = accuracies.std()

print("Accuracy is "+ str(round(accmean*100, 3)) +"%")

Xfi = X[0:]
XColNames = X.columns.values.tolist()
Xfi.columns = XColNames[0:]
features = Xfi.columns
importances = classifier.feature_importances_
indices = np.argsort(importances)

plt.title('LOL Diamond Matches Feature Importances')
plt.barh(range(len(indices)),importances[indices], color = 'r',
         align = 'center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
print("Feature importances defined by a model wth accucracy of " + \
      str(round(accmean*100,3)) + "%")
    
    

#PCA:
from sklearn.decomposition import PCA

pcX_train, pcX_test, pcy_train, pcy_test = train_test_split(X, y,
                                                            test_size = 0.20,
                                                            random_state = 0)

pca = PCA(n_components = 11) 
pcX_train = pca.fit_transform(X_train)
pcX_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

#Fitting the Logistic Regression:
from sklearn.linear_model import LogisticRegression
pcclassifier = LogisticRegression(random_state = 0, solver='liblinear',
                                multi_class='ovr')
pcclassifier.fit(pcX_train, pcy_train)

#predicting test set results:
pcy_pred = pcclassifier.predict(pcX_test) 

#evaluate:
pccm = confusion_matrix(pcy_test, pcy_pred)

# Applying k-Fold Cross Validation
pcaccuracies = cross_val_score(estimator = pcclassifier, X = pcX_train,
                               y = pcy_train, cv = 5)
pcaccmean = pcaccuracies.mean()
pcstddev = pcaccuracies.std()

print("PCA Accuracy is "+ str(round(pcaccmean*100, 3)) +"%")


