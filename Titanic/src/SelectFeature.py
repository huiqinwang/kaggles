# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17-11-1 下午2:15
# @Author  : huiqin
# @Email   : huiqin92@163.com
# @File    : SelectFeature.py
# @Software: PyCharm Community Edition
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import PreprocessData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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

def featureSelect_RandomForest(train_data,test_data):
    '''
    随机森林训练数据集，获得重要特征
    '''
    test_pid = test_data['PassengerId'].reset_index()
    train_data = train_data.drop(['PassengerId'],axis=1)
    test_data = test_data.drop(['PassengerId'], axis=1)
    features_list = train_data.columns.values[1::]

    X = train_data.values[:,1:]
    y = train_data.values[:,0]
    survived_weight = .75
    y_weights = np.array([survived_weight if s == 0 else 1 for s in y])

    print("Rough fitting RandomForest to determine feature importance...")
    forest = RandomForestClassifier(oob_score=False,n_estimators=10)
    forest.fit(X,y,sample_weight=y_weights)


    feature_importance = forest.feature_importances_
    feature_importance = 100.0 * (feature_importance/feature_importance.max())

    fi_threshold = 18
    importance_idx = np.where(feature_importance > fi_threshold)[0]
    importance_features = features_list[importance_idx]
    print("\n",importance_features.shape[0],"importance features(>",fi_threshold,"% of max importance)...\n")

    sorted_idx = np.argsort(feature_importance[importance_idx])[::-1]
    pos = np.arange(sorted_idx.shape[0])+.5
    print(len(importance_idx),len(sorted_idx))
    plt.subplot(1,2,2)
    plt.title("Feature Importance")
    plt.barh(pos,feature_importance[importance_idx][sorted_idx[::-1]],color='r',align='center')
    plt.yticks(pos,importance_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    # plt.draw()
    # plt.show()

    X = X[:,importance_idx][:,sorted_idx]
    # print(X[:5])
    test_data = test_data.iloc[:,importance_idx].iloc[:,sorted_idx]
    # print(test_data.head())
    # print(importance_idx)
    # print(sorted_idx)
    return importance_idx,sorted_idx


if __name__ == "__main__":
    print("test")
    train_data, test_data, origin_data = PreprocessData.process_data(bins=True, scaled=True, binary=True)
    featureSelect_RandomForest(train_data,test_data)