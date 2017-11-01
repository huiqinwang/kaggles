# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17-11-1 上午10:25
# @Author  : huiqin
# @Email   : huiqin92@163.com
# @File    : PreprocessData.py
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
    origin_data['Embarked'] = origin_data['Embarked'].fillna(origin_data.Embarked.dropna().mode().values[0])
    origin_data.Cabin[origin_data.Cabin.isnull()] = 'U0'

    # predict age
    origin_data = predictAge(origin_data)

    # discretization characteristic
    origin_data = discretizationData(origin_data)
    print(origin_data[:10])

    # scaled data
    if keep_scaled:
        origin_data = standardData(origin_data)


def predictAge(origin_data):
    '''
    随机森林回归预测缺失性别
    :param origin_data:
    :return:
    '''
    age_df = origin_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_nn = age_df.loc[origin_data.Age.notnull()]
    age_df_in = age_df.loc[origin_data.Age.isnull()]
    X = age_df_nn.values[:,1:]
    Y = age_df_nn.values[:,0]
    rfr = RandomForestRegressor(n_estimators=1000,n_jobs=1)
    rfr.fit(X,Y)
    predictAges = rfr.predict(age_df_in.values[:,1:])
    origin_data.loc[origin_data.Age.isnull(),'Age'] = predictAges
    return  origin_data

def discretizationData(origin_data):
    '''
    离散化连续数值
    :param origin_data:
    :return:
    '''
    dummies_df = pd.get_dummies(origin_data.Embarked)
    dummies_df = dummies_df.rename(columns=lambda x:"Emarked_"+str(x))
    origin_data['CabinLetter'] = origin_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    origin_data['CabinLetter'] = pd.factorize(origin_data.CabinLetter)[0]
    origin_data['CabinNumber'] = origin_data['Cabin'].map(lambda x: getCabinNumber(x)).astype(int)+1
    origin_data = processFare(origin_data)
    return origin_data

def getCabinNumber(cabin):
    '''
    正则处理船舱编号
    :param cabin:
    :return:
    '''
    match = re.compile("([0-9]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 0

def processFare(origin_data):
    '''
    四分位离散化票价
    :param origin_data:
    :return:
    '''
    origin_data['Fare'][origin_data.Fare.isnull()] = origin_data.Fare.dropna().mean()
    origin_data['Fare'][np.where(origin_data['Fare']==0)[0]] = origin_data['Fare'][origin_data.Fare.nonzero()[0]].min()/10
    origin_data['Fare_bin'] = pd.qcut(origin_data.Fare,4)
    origin_data["Fare_bin_id"] = pd.factorize(origin_data.Fare_bin)[0]+1
    return  origin_data

def standardData(origin_data):
    '''
    数据标准化
    :param origin_data:
    :return:
    '''
    scaler = StandardScaler()
    Fare_array = np.array(origin_data.Fare_bin_id).reshape(len(origin_data.Fare_bin_id),1)
    origin_data['Fare_bin_id_scaled'] = scaler.fit_transform(Fare_array).reshape(len(origin_data.Fare_bin_id),)
    Age_array = np.array(origin_data.Age).reshape(len(origin_data.Age),1)
    origin_data['Age_Scaled'] = scaler.fit_transform(Age_array).reshape(len(origin_data.Age),)
    return origin_data

if __name__ == "__main__":
    dataSet = load_dataSet()
    keep_scaled = True
    preprocess_data(dataSet,keep_scaled)
