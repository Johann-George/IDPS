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