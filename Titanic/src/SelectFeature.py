# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17-11-1 下午2:15
# @Author  : huiqin
# @Email   : huiqin92@163.com
# @File    : SelectFeature.py
# @Software: PyCharm Community Edition
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

global origin_data

def pca():
    '''
    pca降维
    :param origin_data:
    :return:
    '''
    X = origin_data[:,1::]
    y = origin_data.values[:,0]
    variance_pca = .99

    pca = PCA(n_components=variance_pca)
    X_transformed = pca.fit_transform(X,y)
    pca_df = pd.DataFrame(X_transformed)



def dropBySpearman():
    '''
    利用特征之间相关性进行特征选择
    :return:
    '''
    global origin_data
    df_corr = origin_data.drop(['Survived','PassengerId'],axis=1).corr(method='spearman')
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask*df_corr

    drops = []
    for col in df_corr.columns.values:
        if np.in1d([col],drops):
            continue
        corr = df_corr[abs(df_corr[col]) > 0.98].index
        drops = np.union1d(drops,corr)
    print("Dropping", drops.shape[0], 'highly correlated features ...\n')
    origin_data.drop(drops,axis=1,inplace=True)


