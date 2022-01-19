import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class DataCleaning:
    def __init__(self):
        self.result = 0

    def readFile(self):
        df = pd.read_csv("train.csv")
        df.drop(labels=['Cabin', 'Age', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True) #inplace-To replace the file
        df.dropna(axis=0, inplace=True)
        df['Sex'] = df['Sex'].map({'male':0, 'female':1})
        target = df['Survived']
        df.drop(labels=['Survived', 'Embarked'],axis=1, inplace=True)
        return df, target

    def LogisticRegression(self, df, target):
        train_input, test_input, train_target, test_target = train_test_split(
            df, target, random_state=42)
        lr = LogisticRegression()
        lr.fit(train_input, train_target)
        result2 = lr.predict(test_input)
        print(result2)
        print(df.head(0))
        print(lr.coef_)

if __name__ == "__main__":
    a = DataCleaning()
    result = a.readFile()
    a.LogisticRegression(result[0], result[1])
    #test.csv download 후 다시 해보기.