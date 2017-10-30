from sklearn.model_selection import cross_val_score
import pandas as pd

import pandas as pd
root = "/home/wanghuiqin/workspace/data/Titanic/"
origion = pd.read_csv(root+"train.csv")

# split train and test
train_data = origion.drop(['PassengerId','Survived'],axis=1)
# print(train_data.head())

target_data = origion['Survived']
# print(target_data.head())

