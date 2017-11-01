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

def preprocess_data(keep_scaled):
    '''
    对源数据集进行预处理
    :param dataSet:
    :return:
    '''
    origin_data = load_dataSet()

    # fill null
    origin_data['Embarked'] = origin_data['Embarked'].fillna(origin_data.Embarked.dropna().mode().values[0])
    origin_data['Cabin'] = origin_data['Cabin'].fillna("U0")

    # predict age
    print("predict age:                         ")
    origin_data = predictAge(origin_data)

    # discretization characteristic
    print("discretization characteristic:                         ")
    origin_data = discretizationData(origin_data)
    origin_data = qcutData(origin_data)

    # split name
    print("split name:                         ")
    origin_data = splitName(origin_data)

    # preprocess Ticket
    print("preprocess Ticket:                         ")
    origin_data = preprocessTicket(origin_data)

    # scaled data
    print("keep_scaled:                         ")
    if keep_scaled:
        origin_data = standardData(origin_data)

    # combine data
    print("combine data:                         ")
    origin_data = combineFeature(origin_data)
    print(origin_data[:5])

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

def qcutData(origin_data):
    '''
    四分位离散化票价
    :param origin_data:
    :return:
    '''
    feature_list = ['Fare','Age']
    for name in feature_list:
        origin_data[name] = origin_data[name].fillna( origin_data[name].dropna().mean())
        origin_data[name][np.where(origin_data[name]==0)[0]] = origin_data[name][origin_data[name].nonzero()[0]].min()/10
        origin_data[name+"_bin"] = pd.qcut(origin_data[name],4)
        origin_data[name+"_bin_id"] = pd.factorize(origin_data[name+"_bin"])[0]+1
    return  origin_data

def standardData(origin_data):
    '''
    数据标准化
    :param origin_data:
    :return:
    '''
    scaler = StandardScaler()
    feature_list = ['Age', 'Fare', 'Pclass', 'Parch', 'SibSp',
                            'Title_id', 'CabinNumber', 'Age_bin_id', 'Fare_bin_id']
    for name in feature_list:
        origin_data = standardSingle(origin_data,name,scaler)
    return origin_data

def standardSingle(origin_data,name,scaler):
    print(name)
    tmp_array = np.array(origin_data[name]).reshape(len(origin_data[name]),1)
    origin_data[name+"_scaled"] = scaler.fit_transform(tmp_array).reshape(len(origin_data[name]), )
    return origin_data

def splitName(origin_data):
    '''
    分割名字
    :param origin_data:
    :return:
    '''
    origin_data['Title'] = origin_data['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
    origin_data['Title'][origin_data.Title=='Jonkheer'] = 'Master'
    origin_data['Title'][origin_data.Title.isin(['Ms','Mlle'])] = 'Miss'
    origin_data['Title'][origin_data.Title == 'Mme'] = 'Mrs'
    origin_data['Title'][origin_data.Title.isin(['Capt','Don','Major','col','Sir'])] = 'Sir'
    origin_data['Title'][origin_data.Title.isin(['Dona','Lady','the Countess'])] = 'Lady'
    origin_data['Title_id'] = pd.factorize(origin_data.Title)[0]+1
    return  origin_data

def preprocessTicket(origin_data):
    '''
    处理票单号
    :param origin_data:
    :return:
    '''
    origin_data['TicketPreficx'] = origin_data['Ticket'].map(lambda x: getTicketPrefix(x.upper()))
    origin_data['TicketPreficx'] = origin_data['TicketPreficx'].map(lambda x:re.sub('[\.?\/?]','',x))
    origin_data['TicketPreficx'] = origin_data['TicketPreficx'].map(lambda x:re.sub('STON','SOTON',x))
    origin_data['TicketPreficx'] = pd.factorize(origin_data['TicketPreficx'])[0]

    origin_data['TicketNumber'] = origin_data['Ticket'].map(lambda x: getTicketNumber(x))
    origin_data['TicketNumberLength'] = origin_data['TicketNumber'].map(lambda x: len(x)).astype(int)
    origin_data['TicketNumberStart'] = origin_data['TicketNumber'].map(lambda x: x[0:1]).astype(int)
    origin_data['TicketNumber'] = origin_data['TicketNumber'].astype(int)
    return origin_data

def getTicketPrefix(ticket):
    '''
    分割票价前缀
    :param ticket:
    :return:
    '''
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:
        return  match.group()
    else:
        return 'U'

def getTicketNumber(ticket):
    '''
    票号数字部分
    :param ticket:
    :return:
    '''
    match = re.compile("(^[0-9]+$)").search(ticket)
    if match:
        return  match.group()
    else:
        return  '0'


def combineFeature(origin_data):
    '''
    组合特征
    :param origin_data:
    :return:
    '''
    numerics = origin_data.loc[:,['Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', 'SibSp_scaled',
                            'Title_id_scaled', 'CabinNumber_scaled', 'Age_bin_id_scaled', 'Fare_bin_id_scaled']]
    new_fields_count = 0
    for i in range(numerics.columns.size-1):
        for j in range(numerics.columns.size-1):
            if i <= j:
                name = str(numerics.columns.values[i]) + "*" + str(numerics.columns.values[j])
                origin_data = pd.concat([origin_data,pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j],name=name)],axis=1)
                new_fields_count +=1
            if i < j:
                name = str(numerics.columns.values[i]) + "+" + str(numerics.columns.values[j])
                origin_data = pd.concat([origin_data,pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j],name=name)],axis=1)
                new_fields_count +=1
            if not i == j:
                name = str(numerics.columns.values[i]) + "/" + str(numerics.columns.values[j])
                origin_data = pd.concat([origin_data,pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j],name=name)],axis=1)
                new_fields_count +=1
                name = str(numerics.columns.values[i]) + "-" + str(numerics.columns.values[j])
                origin_data = pd.concat([origin_data,pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j],name=name)],axis=1)
                new_fields_count +=1
    print("\n",new_fields_count,"new features generated")
    return  origin_data


if __name__ == "__main__":
    keep_scaled = True
    preprocess_data(keep_scaled)
