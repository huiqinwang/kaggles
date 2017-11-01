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


class featureRandomForest(df):
    '''
    随机森林训练数据集，获得重要特征
    '''
    X = df.values[:,1:]
    y = df.values[:,0]
    survived_weight = .75
    y_weights = np.array([survived_weight if s == 0 else 1 for s in y])

    print("Rough fitting RandomForest to determine feature importance...")
    forest = RandomForestClassifier(oob_score=True,n_estimators=10000)
    forest.fit(X,y,sample_weight=y_weights)
    feature_importance = forest.feature_importances_
    feature_importance = 100.0 * (feature_importance/feature_importance.max())

    fi_threshold = 18
    importance_idx = np.where(feature_importance > fi_threshold)[0]
    importance_features = features_list[importance_idx]
    print("\n",importance_features.shape[0],"importance features(>",fi_threshold,"% of max importance)...\n")

    sorted_idx = np.argsort(feature_importance[importance_idx])[::-1]
    pos = np.arrange(sorted_idx.shape[0])+.5
    plt.subplot(1,2,2)
    plt.title("Feature Importance")
    plt.barh(pos,feature_importance[importance_idx][sorted_idx[::-1]],color='r',align='center')
    plt.yticks(pos,importance_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.draw()
    plt.show()
    X = X[:,importance_idx][:,sorted_idx]
    submit_df = submit_df.iloc[:,importance_idx].iloc[:,sorted_idx]



def selectParameter(X,y):
    '''
    模型参数选择
    :param df:
    :return:
    '''
    sqrtFeat = int(np.sqrt(X.shape[1]))
    minSamSplit = int(X.shape[0]*0.015)

    grid_test1 = {'n_estimators'    :[1000,2500,5000],
                  'criterion'       :['gini','entropy'],
                  'max_feature'     :[sqrtFeat-1,sqrtFeat,sqrtFeat+1],
                  'max_depth'       :[5,10,25],
                  'min_samples_split'  :[2,5,10,minSamSplit]}
    forest = RandomForestClassifier(oob_score=True)
    print("Hyperparameter optimization using GridSearchCV...")
    grid_search = GridSearchCV(forest,grid_test1,n_jobs=-1,cv=10)
    grid_search.fit(X,y)
    best_params_from_grid_search = report(grid_search.grid_scores_)



def report(grid_scores,n_top=5):
    '''
    输出模型选择后结果
    :param grid_scores:
    :param n_top:
    :return:
    '''
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
