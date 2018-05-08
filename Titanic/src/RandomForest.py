# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17-11-1 下午3:16
# @Author  : huiqin
# @Email   : huiqin92@163.com
# @File    : RandomForest.py
# @Software: PyCharm Community Edition
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import PreprocessData
import SelectFeature
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def randomForestTrain(train_data,test_data,importance_idx=None,sorted_idx=None):
    '''
    随机森林预测
    :return:
    '''
    test_pid = test_data['PassengerId'].reset_index()
    train_data = train_data.drop(['PassengerId'], axis=1)
    test_data = test_data.drop(['PassengerId'], axis=1)

    X = train_data.values[:, 1:]
    y = train_data.values[:, 0]
    # choose best feature before 0.8097
    # X = X[:, importance_idx][:, sorted_idx]
    # test_data = test_data.iloc[:,importance_idx].iloc[:,sorted_idx]
    # X_test = test_data
    # print(test_data.head())

    # split
    # train_X,test_X,train_y,test_y = train_test_split(X,y,random_state=42,test_size=0.2)
    # select best parameter
    # best_parameter = selectParameter(train_X,train_y)
    # print(best_parameter)

    sqrtFeat = int(np.sqrt(X.shape[1]))
    minSamSplit = int(X.shape[0] * 0.015)

    forest = RandomForestClassifier(
                 n_estimators=10000,
                 criterion="gini",
                 max_depth=30,
                 min_samples_split=minSamSplit,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=True,
                 n_jobs=3,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None)

    # 给标签加权重，解决不平衡问题
    survived_weight = .75
    # y_weights = np.array([survived_weight if s == 0 else 1 for s in train_y])
    # forest.fit(train_X, train_y, sample_weight=y_weights)
    y_weights = np.array([survived_weight if s == 0 else 1 for s in y])
    forest.fit(X, y, sample_weight=y_weights)

    # acurracy
    print(forest.oob_score_)
    # predict_y = forest.predict(test_X)
    # print(accuracy_score(test_y,predict_y))
    # print(confusion_matrix(test_y,predict_y))

    # forest
    X_test = test_data.values[:, 1:]
    y_predict = forest.predict(X_test)

    y_series = pd.Series(y_predict, name='Survived').map(lambda x: int(x))
    print(y_series[-1::])

    result = pd.concat([test_pid, y_series], axis=1).drop('index', axis=1)
    print(result[:5])
    files = open("/Users/huiqin08/WorkSpace/Git_WorkSpace/data/Titanic/gender_submission.csv", mode='wb')
    result.to_csv(files, index=False)

def selectParameter(X,y):
    '''
    模型参数选择
    :param df:
    :return:
    '''
    sqrtFeat = int(np.sqrt(X.shape[1]))
    minSamSplit = int(X.shape[0]*0.015)

    # grid_test1 = {'n_estimators'    :[1000,2500,5000],
    #               'criterion'       :['gini','entropy'],
    #               'max_feature'     :[sqrtFeat-1,sqrtFeat,sqrtFeat+1],
    #               'max_depth'       :[5,10,25],
    #               'min_samples_split'  :[2,5,10,minSamSplit]}
    grid_test1 = {'n_estimators'    :[1000,10],
                  'criterion'       :['gini','entropy'],
                  'max_features'     :[sqrtFeat-1,sqrtFeat],
                  'max_depth'       :[5,10],
                  'min_samples_split'  :[2,5]}
    forest = RandomForestClassifier(oob_score=True)
    print("Hyperparameter optimization using GridSearchCV...")
    grid_search = GridSearchCV(forest,grid_test1,n_jobs=3,cv=10)
    grid_search.fit(X,y)
    best_params_from_grid_search = report(grid_search.grid_scores_)
    return best_params_from_grid_search

def report(grid_scores,n_top=5):
    '''
    输出模型选择后结果
    :param grid_scores:
    :param n_top:
    :return:
    '''
    print(grid_scores)
    print(type(grid_scores))
    params = None
    top_scores = sorted(grid_scores,key=itemgetter(1),reverse=True)[::n_top]
    for i,score in enumerate(top_scores):
        print("Parameters with rank: {0}".format(i+1))
        print("Mean Validation scores: {0:.4f} (std: {1:.4f})").format(score.mean_validation_score,np.std(score.cv_validation_scores))
        print("Parameters: {0}".format(score.parameters))
        print("")

        if params == None:
            params = score.parameters
    return params


if __name__ == "__main__":
    print("test")
    train_data,test_data,origin_data = PreprocessData.process_data(bins=True,scaled=True,binary=True)
    print(len(train_data.columns))
    # featureRandomForest(train_data,test_data)
    # importance_idx,sorted_idx = SelectFeature.featureSelect_RandomForest(train_data,test_data)
    randomForestTrain(train_data,test_data)
