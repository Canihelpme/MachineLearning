import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class DataCleaning:
    def __init__(self):
        self.result = 0

    def readFile(self):
        df = pd.read_csv("train.csv")
        df.drop(labels=['Cabin', 'Age', 'Name', 'Ticket'], axis=1, inplace=True) #inplace-To replace the file
        df.dropna(axis=0, inplace=True)
        df['Sex'] = df['Sex'].map({'male':0, 'female':1})
        target = df['Survived']
        df.drop(labels=['Survived', 'Embarked'],axis=1, inplace=True)
        return df, target

    def LogisticRegression(self, df, target):
        test_input = train_test_split(df, target, random_state=42)
        lr = LogisticRegression()
        lr.fit(df, target)
        result2 = lr.predict(test_input)
        #print(result2)
        print(lr.coef_)

a = DataCleaning()
result = a.readFile()
a.LogisticRegression(result[0], result[1])