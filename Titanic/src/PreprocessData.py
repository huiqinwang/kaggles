# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17-11-1 上午10:25
# @Author  : huiqin
# @Email   : huiqin92@163.com
# @File    : processData.py
# @Software: PyCharm Community Edition
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import re

np.set_printoptions(precision=3,threshold=10000,linewidth=160,edgeitems=999,suppress=True)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',160)
pd.set_option('expand_frame_repr',False)
pd.set_option('precision',4)


def load_dataSet():
    '''
    获取源数据集
    :return: dataFrame
    '''
    root_path = '/home/wanghuiqin/workspace/data/Titanic/'
    train_data = pd.read_csv(root_path+"train.csv",header=0)
    test_data = pd.read_csv(root_path+"test.csv",header=0)
    dataSet = pd.concat([train_data,test_data])
    dataSet.reset_index(inplace=True)
    dataSet.drop(['index'],axis=1,inplace=True)
    dataSet = dataSet.reindex(train_data.columns,axis=1)
    print(len(train_data),len(test_data))
    return train_data,test_data,dataSet

def process_data(binary=False, bins=False, scaled=False, strings=False,
                    raw=True):
    '''
    对源数据集进行预处理
    :param dataSet:
    :return:
    '''
    global keep_binary,keep_bins,keep_scaled,keep_strings,keep_raw,origin_data
    keep_binary = binary
    keep_bins = bins
    keep_scaled = scaled
    keep_strings = strings
    keep_raw = raw
    train_data,test_data,origin_data = load_dataSet()

    # process feature
    processCabin()
    processTicket()
    processName()
    processFare()
    processEmbarked()
    processFamily()
    processSex()
    processPClass()
    processAge()
    processDrops()

    # combine feature
    combineFeature()

    # change survived index
    columns_list = list(origin_data.columns.values)
    columns_list.remove('Survived')
    new_col_list = list()
    new_col_list.append('Survived')
    new_col_list.extend(columns_list)

    origin_data = origin_data.reindex(columns=new_col_list)
    print("feature's lens: ", origin_data.columns.size,len(origin_data.columns), "manually generated features...\n", origin_data.columns.values)

    train_data = origin_data[:train_data.shape[0]]
    test_data = origin_data[train_data.shape[0]:]
    print(train_data[:5])
    print(test_data[:5])
    print(len(train_data), len(test_data))

    return train_data,test_data,origin_data


def processCabin():
    '''
    船舱处理：fillna split scaled
    :return:
    '''
    global origin_data
    origin_data['Cabin'] = origin_data['Cabin'].fillna("U0")
    origin_data['CabinLetter'] = origin_data['Cabin'].map(lambda x: getRepr(x,"([a-zA-Z]+)",'U'))
    origin_data['CabinLetter'] = pd.factorize(origin_data.CabinLetter)[0]

    if keep_binary:
        cabin_dummy = pd.get_dummies(origin_data['CabinLetter']).rename(columns=lambda x: 'CabinLetter_'+str(x))
        origin_data = pd.concat([origin_data,cabin_dummy],axis=1)

    origin_data['CabinNumber'] = origin_data['Cabin'].map(lambda x: getRepr(x,"([0-9]+)",0)).astype(int)+1
    if keep_scaled:
        singleStandard('CabinNumber')

def processTicket():
    '''
    处理票单号
    :param origin_data:
    :return:
    '''
    global origin_data
    origin_data['TicketPrefix'] = origin_data['Ticket'].map(lambda x: getRepr(x.upper(),"([a-zA-Z\.\/]+)",'U'))
    origin_data['TicketPrefix'] = origin_data['TicketPrefix'].map(lambda x:re.sub('[\.?\/?]','',x))
    origin_data['TicketPrefix'] = origin_data['TicketPrefix'].map(lambda x:re.sub('STON','SOTON',x))
    origin_data['TicketPrefix_id'] = pd.factorize(origin_data['TicketPrefix'])[0]

    if keep_binary:
        ticket_dummy = pd.get_dummies(origin_data['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_'+str(x))
        origin_data = pd.concat([origin_data,ticket_dummy],axis=1)
    origin_data.drop(['TicketPrefix'],axis=1, inplace=True)

    origin_data['TicketNumber'] = origin_data['Ticket'].map(lambda x: getRepr(x,"(^[0-9]+$)",'0'))
    origin_data['TicketNumberLength'] = origin_data['TicketNumber'].map(lambda x: len(x)).astype(int)
    origin_data['TicketNumberStart'] = origin_data['TicketNumber'].map(lambda x: x[0:1]).astype(int)
    origin_data['TicketNumber'] = origin_data['TicketNumber'].astype(int)

    if keep_scaled:
        singleStandard('TicketNumber')

def processName():
    '''
    处理名称
    :return:
    '''
    global origin_data
    origin_data['Names'] = origin_data['Name'].map(lambda x: len(re.split(' ',x)))

    origin_data['Title'] = origin_data['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])

    index_1 = origin_data.Title=='Jonkheer'
    origin_data.loc[index_1,'Title']= 'Master'
    origin_data.loc[origin_data.Title.isin(['Ms','Mlle']),'Title']= 'Miss'
    origin_data.loc[origin_data.Title == 'Mme','Title'] = 'Mrs'
    origin_data.loc[origin_data.Title.isin(['Capt','Don','Major','col','Sir']),'Title'] = 'Sir'
    origin_data.loc[origin_data.Title.isin(['Dona','Lady','the Countess']),'Title'] = 'Lady'


    if keep_binary:
        title_dummy = pd.get_dummies(origin_data['Title']).rename(columns=lambda x: 'Title_'+str(x))
        origin_data = pd.concat([origin_data,title_dummy],axis=1)

    if keep_scaled:
        singleStandard('Names')

    if keep_bins:
        origin_data['Title_id'] = pd.factorize(origin_data.Title)[0] + 1

    if keep_bins and keep_scaled:
        singleStandard('Title_id')


def processFare():
    '''
    处理票费
    :return:
    '''
    global origin_data

    origin_data.loc[np.isnan(origin_data['Fare']),'Fare'] =  origin_data['Fare'].median()
    origin_data.loc[np.where(origin_data['Fare'] == 0)[0],'Fare']= origin_data.loc[origin_data['Fare'].nonzero()[0],'Fare'].min() / 10
    origin_data['Fare_bin'] = pd.qcut(origin_data['Fare'], 4)

    if keep_binary:
        origin_data = pd.concat([origin_data, pd.get_dummies(origin_data['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))],axis=1)

    if keep_bins:
        origin_data['Fare_bin_id'] = pd.factorize(origin_data['Fare_bin'])[0] + 1

    if keep_scaled:
        singleStandard('Fare')

    if keep_bins and keep_scaled:
        singleStandard('Fare_bin_id')

    if not keep_strings:
        origin_data.drop('Fare_bin', axis=1, inplace=True)

def singleStandard(name):
    '''
    单特征标准化
    :return:
    '''
    scaler = StandardScaler()
    global origin_data
    origin_data[name+"_scaled"] = np.ravel(scaler.fit_transform(np.array(origin_data[name]).reshape(-1,1)))


def processEmbarked():
    '''
    处理登陆点
    :return:
    '''
    global origin_data
    origin_data.loc[origin_data.Embarked.isnull(),'Embarked']= origin_data.Embarked.dropna().mode().values
    origin_data['Embarked'] = pd.factorize(origin_data['Embarked'])[0]
    if keep_binary:
        origin_data = pd.concat([origin_data, pd.get_dummies(origin_data['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))],axis=1)


def processPClass():
    '''
    处理等级
    :return:
    '''
    global origin_data
    origin_data.loc[origin_data.Pclass.isnull(),'Pclass'] = origin_data.Pclass.dropna().mode().values

    if keep_binary:
        origin_data = pd.concat([origin_data, pd.get_dummies(origin_data['Pclass']).rename(columns=lambda x: 'Pclass_' + str(x))],axis=1)

    if keep_scaled:
        singleStandard('Pclass')


def processFamily():
    '''
    处理家庭状况
    :return:
    '''
    global origin_data
    origin_data['SibSp'] = origin_data['SibSp'] + 1
    origin_data['Parch'] = origin_data['Parch'] + 1
    if keep_scaled:
        singleStandard('SibSp')
        singleStandard('Parch')

    if keep_binary:
        sibsps = pd.get_dummies(origin_data['SibSp']).rename(columns=lambda x: 'SibSp_' + str(x))
        parchs = pd.get_dummies(origin_data['Parch']).rename(columns=lambda x: 'Parch_' + str(x))
        origin_data = pd.concat([origin_data, sibsps, parchs], axis=1)

def processSex():
    '''
    处理性别
    :return:
    '''
    global origin_data
    origin_data['Gender'] = np.where(origin_data['Sex'] == 'male', 1, 0)


def getRepr(name,repr,rets):
    '''
    正则表达式截取结果
    :param name: 特征
    :param repr: 表达式
    :param rets: 未匹配结果
    :return: 匹配结果
    '''
    match = re.compile(repr).search(name)
    if match:
        return match.group()
    else:
        return rets


def processAge():
    '''

    :return:
    '''
    global origin_data
    setMissingAges()

    # center the mean and scale to unit variance
    if keep_scaled:
        singleStandard('Age')

    # have a feature for children
    origin_data['isChild'] = np.where(origin_data.Age < 13, 1, 0)

    # bin into quartiles and create binary features
    origin_data['Age_bin'] = pd.qcut(origin_data['Age'], 4)
    if keep_binary:
        origin_data = pd.concat([origin_data, pd.get_dummies(origin_data['Age_bin']).rename(columns=lambda x: 'Age_' + str(x))],axis=1)

    if keep_bins:
        origin_data['Age_bin_id'] = pd.factorize(origin_data['Age_bin'])[0] + 1

    if keep_bins and keep_scaled:
        singleStandard('Age_bin_id')

    if not keep_strings:
        origin_data.drop('Age_bin', axis=1, inplace=True)

def setMissingAges():
    '''
    随机森林回归预测缺失性别
    :param origin_data:
    :return:
    '''
    global origin_data

    age_df = origin_data[['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title_id','Pclass','Names','CabinLetter']]
    age_df_nn = age_df.loc[origin_data.Age.notnull()]
    age_df_in = age_df.loc[origin_data.Age.isnull()]
    X = age_df_nn.values[:,1::]
    Y = age_df_nn.values[:,0]
    rfr = RandomForestRegressor(n_estimators=1000,n_jobs=1)
    rfr.fit(X,Y)
    predictAges = rfr.predict(age_df_in.values[:,1:])
    origin_data.loc[origin_data.Age.isnull(),'Age'] = predictAges

def processDrops():
    '''
    删除列表
    :return:
    '''
    global origin_data
    rawDropList = ['Name', 'Names', 'Title', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked', \
                   'Cabin', 'CabinLetter', 'CabinNumber', 'Age', 'Fare', 'Ticket', 'TicketNumber']
    stringsDropList = ['Title', 'Name', 'Cabin', 'Ticket', 'Sex', 'Ticket', 'TicketNumber']

    if not keep_raw:
        origin_data.drop(rawDropList, axis=1, inplace=True)
    elif not keep_strings:
        origin_data.drop(stringsDropList, axis=1, inplace=True)


def combineFeature():
    '''
    组合特征
    :param origin_data:
    :return:
    '''
    global origin_data

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



if __name__ == "__main__":
    train,test,result = process_data(bins=True,scaled=True,binary=True)
    from sklearn.model_selection import KFold
    n = KFold()