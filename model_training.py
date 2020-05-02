from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE

x_train = pd.read_csv('./files/x_train.csv')
y_train = pd.read_csv('./files/y_train.csv')
x_test = pd.read_csv('./files/x_test.csv')
y_test = pd.read_csv('./files/y_test.csv')

def make_and_save_decision_tree_model():
    dt = GridSearchCV(DecisionTreeRegressor(), {'max_depth': [4,5,6,7], 'min_samples_leaf': [0.05, 0.1, 0.15, 0.2, 0.3] }, cv=5, return_train_score=False)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)

    mse_dt = MSE(y_test, y_pred)
    rmse_dt = mse_dt**(1/2)

    print('mse:', mse_dt)
    print('rmse:' , rmse_dt)
    print('best score:', dt.best_score_)
    print('best param:', dt.best_params_)

    df = pd.DataFrame(dt.cv_results_)
    print(df)

    pd.DataFrame(dt.cv_results_).to_pickle('./decision_tree_model.pkl')

def make_svr_model():
    #{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1,2,3,4,5,6,7,8,9,10]}
    model = GridSearchCV(SVC(), {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1,2,3,4,5,6,7,8,9,10]}, cv=5, return_train_score=False)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse_dt = MSE(y_test, y_pred)
    rmse_dt = mse_dt ** (1 / 2)

    print('rmse:', rmse_dt)
    print('best score:', model.best_score_)
    print('best param:', model.best_params_)

    model.fit(x_train, y_train)
    model.score(x_test, y_test)

    pd.DataFrame(model.cv_results_).to_pickle('./decision_tree_model.pkl')

#make_and_save_decision_tree_model()
make_svr_model()


#cross_val_score(svm.SVC(kernel='rbf', C=10, gamma='auto'), iris.data, iris.target, cv=10)

''' ezt helyettesíti a gridsearchCV
kernels=['rbf', 'linear']
c = [1,10,20]
avg_scores = {}
for kval in kernels:
    for cval in c:
        cv_scores = cross_val_score(svm.SVC(kernel=kval, C=cval, gamma='auto'), iris.data, iris.target, cv=5)
        avg_scores[kval + '_' + str(cval)] = np.average(cv_scores)
    
 avg_scores   
'''

'''
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(svmSVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
    }, cv=5, return_train_score=False)
    
clf.fit(iris.data, iris.target)
clf.cv_results_

df = pd.DataFrame(clf.cv_results_)
df

df[['param_C', 'param_kernel', 'mean_test_score']]
}

#dir(clf) - kiiratás

clf.best_score_

clf.best_params_

'''