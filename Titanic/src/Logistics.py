from sklearn.model_selection import cross_val_score
import pandas as pd

import pandas as pd
root = "/home/wanghuiqin/workspace/data/Titanic/"
origion = pd.read_csv(root+"train.csv")

# string to float
train_data = origion.drop(['PassengerId','Survived'],axis=1)
target_data = origion['Survived']
from sklearn.preprocessing import OneHotEncoder

# split train and test
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(train_data,target_data)

from sklearn.linear_model.logistic import LogisticRegression

lg_model = LogisticRegression()
lg_model.fit(train_x,train_y)
predict_y = lg_model.predict(test_x)

from sklearn.metrics import accuracy_score

result = accuracy_score(test_y,predict_y)
print(result)