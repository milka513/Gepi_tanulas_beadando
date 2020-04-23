from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.pipeline import Pipeline

x_train = pd.read_csv('./files/x_train.csv')
y_train = pd.read_csv('./files/y_train.csv')
x_test = pd.read_csv('./files/x_test.csv')
y_test = pd.read_csv('./files/y_test.csv')

def tanitas():
    print('Start:')
    cimkek = ['neg_mean_squared_error', 'r2']
    for cimke in cimkek:
        tuned_parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}]
        knr = KNeighborsRegressor(weights='distance')
        gridsearch = GridSearchCV(knr, tuned_parameters, scoring=cimke, cv=5, return_train_score=True,refit=True)
        gridsearch.fit(x_train, y_train)
        df = pd.DataFrame(gridsearch.cv_results_)

        print("Best parameters set found on development set:\n")
        print(gridsearch.best_params_)
        print("\nGrid scores on development set:\n")
        print(gridsearch.cv_results_.keys())

        pd.DataFrame(gridsearch.cv_results_).to_pickle('./baseline_model{0}.pkl'.format(cimke))

def test():
    df = pd.read_pickle('./baseline_modelr2.pkl')
    print(df.columns)
    best_of = df[df['rank_test_score'] == 1]
    print(best_of[['param_n_neighbors']])
    print(best_of[['mean_test_score']])
    print(best_of[['mean_train_score']])
    print(best_of[['mean_test_score']])

    base_model = KNeighborsRegressor(n_neighbors=best_of['param_n_neighbors'].values[0], weights='distance')
    base_model.fit(x_train, y_train)
    print(base_model.score(x_test, y_test))


test()