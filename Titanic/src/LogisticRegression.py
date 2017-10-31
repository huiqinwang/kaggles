# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17-10-31 上午10:52
# @Author  : huiqin
# @Email   : huiqin92@163.com
# @File    : LogisticRegression.py
# @Software: PyCharm Community Edition
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import re

def load_dataSet():
    '''
    获取源数据集
    :return: dataFrame
    '''
    root_path = '/home/wanghuiqin/workspace/data/Titanic/'
    dataSet = pd.read_csv(root_path+"train.csv")
    return dataSet

def preprocess_data(dataSet,keep_scaled):
    '''
    对源数据集进行预处理
    :param dataSet:
    :return:
    '''
    origin_data = dataSet

    # fill null
    # origin_data['Age'] = origin_data['Age'].fillna(int(origin_data['Age'].mean()))
    origin_data['Embarked'] = origin_data['Embarked'].fillna(origin_data.Embarked.dropna().mode().values[0])
    origin_data.Cabin[origin_data.Cabin.isnull()] = 'U0'

    # predict age
    age_df = origin_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_nn = age_df.loc[origin_data.Age.notnull()]
    age_df_in = age_df.loc[origin_data.Age.isnull()]
    X = age_df_nn.values[:,1:]
    Y = age_df_nn.values[:,0]
    rfr = RandomForestRegressor(n_estimators=1000,n_jobs=1)
    rfr.fit(X,Y)
    predictAges = rfr.predict(age_df_in.values[:,1:])
    origin_data.loc[origin_data.Age.isnull(),'Age'] = predictAges

    # discretization characteristic
    dummies_df = pd.get_dummies(origin_data.Embarked)
    dummies_df = dummies_df.rename(columns=lambda x:"Emarked_"+str(x))
    origin_data['CabinLetter'] = origin_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    origin_data['CabinLetter'] = pd.factorize(origin_data.CabinLetter)[0]
    origin_data['CabinNumber'] = origin_data['Cabin'].map(lambda x: getCabinNumber(x)).astype(int)+1
    print(origin_data[:10])

    # scaled data
    if keep_scaled:
        scaler = StandardScaler()
        origin_data['Age_Scaled'] = scaler.fit_transform(origin_data['Age'])
    


    # string to oneHotEncoding
    # one_hot = OneHotEncoder(categorical_features=np.array([1,6]))
    # oneHot_data = one_hot.fit_transform(prepro_data)
    # print(oneHot_data[:10])

def getCabinNumber(cabin):
    match = re.compile("([0-9]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 0

if __name__ == "__main__":
    dataSet = load_dataSet()
    preprocess_data(dataSet)
