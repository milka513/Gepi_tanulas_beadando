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

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

x_train = pd.read_csv('./files/x_train.csv')
y_train = pd.read_csv('./files/y_train.csv')
x_test = pd.read_csv('./files/x_test.csv')
y_test = pd.read_csv('./files/y_test.csv')

def tanitas():
    print('Start:')
    cimkek = ['neg_mean_squared_error', 'r2']
    feature_num=x_test.shape[1]
    print(feature_num)
    for cimke in cimkek:
        skb = SelectKBest(f_regression)
        tuned_parameters = [{'skb__k': range(1, feature_num+1),
                            'knr__n_neighbors': range(2, 11)}]

        knr = KNeighborsRegressor(weights='distance')
        model=Pipeline(steps=[('skb', skb), ('knr', knr)])
        gridsearch = GridSearchCV(model, tuned_parameters, scoring=cimke, cv=5, return_train_score=True,refit=True)
        gridsearch.fit(x_train, y_train.values.ravel())
        df = pd.DataFrame(gridsearch.cv_results_)

        print("Best parameters set found on development set:\n")
        print(gridsearch.best_params_)
        print("\nGrid scores on development set:\n")
        print(gridsearch.cv_results_.keys())

        pd.DataFrame(gridsearch.cv_results_).to_pickle('./baseline_model{0}_with_feature_selection.pkl'.format(cimke))

def teszt():
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

def teszt_plot():
    df = pd.read_pickle('./baseline_modelr2_with_feature_selection.pkl')
    print("--------------r2 score-------------")
    best_of = df[df['rank_test_score'] == 1]
    print('n_neighbours száma',best_of['param_knr__n_neighbors'].values[0])
    print('legjobb validacios score',best_of['mean_test_score'].values[0])
    print('hozza tartozo train score',best_of['mean_train_score'].values[0])

    skb = SelectKBest(f_regression, k=best_of['param_skb__k'].values[0])
    knr = KNeighborsRegressor(n_neighbors=best_of['param_knr__n_neighbors'].values[0], weights='distance')
    base_model = Pipeline(steps=[('skb', skb), ('knr', knr)])
    base_model.fit(x_train, y_train.values.ravel())
    y_predict = base_model.predict(x_test)
    print('teszt score', r2_score(y_test, y_predict))
    plt.figure(figsize=(10,10))

    best_k_select=best_of['param_skb__k'].values[0]

    plt.plot(df[df['param_skb__k']==best_k_select]['param_knr__n_neighbors'], df[df['param_skb__k']==best_k_select]['mean_test_score'],'r')
    plt.plot(df[df['param_skb__k']==best_k_select]['param_knr__n_neighbors'], df[df['param_skb__k']==best_k_select]['mean_train_score'],'b')
    plt.show()


    print('------------mean squared error-----------')
    df = pd.read_pickle('./baseline_modelneg_mean_squared_error_with_feature_selection.pkl')
    best_of = df[df['rank_test_score'] == 1]
    print('n_neighbours száma',best_of['param_knr__n_neighbors'].values[0])
    print('legjobb validacios score',best_of['mean_test_score'].values[0])
    print('hozza tartozo train score',best_of['mean_train_score'].values[0])
    plt.figure(figsize=(10,10))

    plt.plot(df[df['param_skb__k'] == best_k_select]['param_knr__n_neighbors'],
             df[df['param_skb__k'] == best_k_select]['mean_test_score'], 'r')
    plt.plot(df[df['param_skb__k'] == best_k_select]['param_knr__n_neighbors'],
             df[df['param_skb__k'] == best_k_select]['mean_train_score'], 'b')
    plt.show()
    skb = SelectKBest(f_regression, k=best_of['param_skb__k'].values[0])
    knr=KNeighborsRegressor(n_neighbors=best_of['param_knr__n_neighbors'].values[0], weights='distance')
    base_model=Pipeline(steps=[('skb', skb), ('knr', knr)])
    base_model.fit(x_train, y_train.values.ravel())
    y_predict=base_model.predict(x_test)
    print('teszt score',mean_squared_error(y_test, y_predict))

teszt_plot()