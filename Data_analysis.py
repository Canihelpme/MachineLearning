import pandas as pd
import numpy as np
import random as ran

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings(action='ignore')

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
total = [train_df, test_df]
by_Pclass = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Pclass')
by_Sex = train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
by_SibSp = train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
by_Parch = train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#groupby를 통해 Pclass와 Survived 묶어서 생존률 판단

#print(train_df.head(50))
#print(train_df.columns.values)

#print(train_df.info()) //각 csv파일 info
#print(test_df.info())

#print(train_df.describe()) #정수형 배열 체크
#print(train_df.describe(include=['O'])) #Obj형 배열 체크

print(by_Pclass)