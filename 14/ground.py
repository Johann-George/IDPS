import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance

df = pd.read_csv('./data/CICIDS2017_sample.csv')

#the code snippet identifies and returns the names of all columns in the DataFrame df that are not of type 'object', such as numerical or datetime columns.
features = df.dtypes[df.dtypes != 'object'].index

#Standardization - Mean 0
df[features] = df[features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# Fill empty values by 0
df = df.fillna(0)

#Transform output to numerical values (last column)
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])


#temporarily remove minor values
df_minor = df[(df['Label']==6)|(df['Label']==1)|(df['Label']==4)]
df_major = df.drop(df_minor.index)


#x=input, y=output
X = df_major.drop(['Label'],axis=1)
y = df_major.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)


from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)

