# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17-11-1 下午2:15
# @Author  : huiqin
# @Email   : huiqin92@163.com
# @File    : SelectFeature.py
# @Software: PyCharm Community Edition
from sklearn.decomposition import PCA
import pandas as pd


def pca(origin_data):
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
