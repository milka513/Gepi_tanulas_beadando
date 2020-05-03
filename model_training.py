from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
import pickle

x_train = pd.read_csv('./files/x_train.csv')
y_train = pd.read_csv('./files/y_train.csv')
x_test = pd.read_csv('./files/x_test.csv')
y_test = pd.read_csv('./files/y_test.csv')

def make_and_save_decision_tree_model():
    dt = GridSearchCV(DecisionTreeRegressor(), {'max_depth': [4,5,6,7], 'min_samples_leaf': [0.05, 0.1, 0.15, 0.2, 0.3] }, cv=5, return_train_score=True)
    dt.fit(x_train, y_train.values.ravel())
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
    model = GridSearchCV(SVC(), {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1,2,3,4,5,6,7,8,9,10]}, cv=5, return_train_score=True)
    model.fit(x_train, y_train.values.ravel())

    y_pred = model.predict(x_test)
    mse_dt = MSE(y_test, y_pred)
    rmse_dt = mse_dt ** (1 / 2)

    print('rmse:', rmse_dt)
    print('best score:', model.best_score_)
    print('best param:', model.best_params_)

    model.score(x_test, y_test)

    pd.DataFrame(model.cv_results_).to_pickle('./svr_model.pkl')

#pca
x_train_dr_pca = pd.read_csv('./files/x_train_pca_reduced.csv')
y_train_dr_pca = pd.read_csv('./files/y_train_pca_reduced.csv')
x_test_dr_pca = pd.read_csv('./files/x_test_pca_reduced.csv')
y_test_dr_pca = pd.read_csv('./files/y_test_pca_reduced.csv')

def make_and_save_decision_tree_model_with_dimred_pca():
    dt = GridSearchCV(DecisionTreeRegressor(),
                      {'max_depth': [4, 5, 6, 7], 'min_samples_leaf': [0.05, 0.1, 0.15, 0.2, 0.3]}, cv=5,
                      return_train_score=True)
    dt.fit(x_train_dr_pca, y_train_dr_pca.values.ravel())
    y_pred = dt.predict(x_test_dr_pca)

    mse_dt = MSE(y_test_dr_pca, y_pred)
    rmse_dt = mse_dt ** (1 / 2)

    print('mse:', mse_dt)
    print('rmse:', rmse_dt)
    print('best score:', dt.best_score_)
    print('best param:', dt.best_params_)

    df = pd.DataFrame(dt.cv_results_)
    print(df)

    pd.DataFrame(dt.cv_results_).to_pickle('./decision_tree_model_pca_reduced.pkl')

def make_svr_model_with_dimred_pca():
    model = GridSearchCV(SVC(), {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1,2,3,4,5,6,7,8,9,10]}, cv=5, return_train_score=True)
    model.fit(x_train_dr_pca, y_train_dr_pca.values.ravel())

    y_pred = model.predict(x_test_dr_pca)
    mse_dt = MSE(y_test_dr_pca, y_pred)
    rmse_dt = mse_dt ** (1 / 2)

    print('rmse:', rmse_dt)
    print('best score:', model.best_score_)
    print('best param:', model.best_params_)

    model.score(x_test_dr_pca, y_test_dr_pca)

    pd.DataFrame(model.cv_results_).to_pickle('./svr_model_pca_reduced.pkl')

#svd
x_train_dr_svd = pd.read_csv('./files/x_train_svd_reduced.csv')
y_train_dr_svd = pd.read_csv('./files/y_train_svd_reduced.csv')
x_test_dr_svd = pd.read_csv('./files/x_test_svd_reduced.csv')
y_test_dr_svd = pd.read_csv('./files/y_test_svd_reduced.csv')

def make_and_save_decision_tree_model_with_dimred_svd():
    dt = GridSearchCV(DecisionTreeRegressor(),
                      {'max_depth': [4, 5, 6, 7], 'min_samples_leaf': [0.05, 0.1, 0.15, 0.2, 0.3]}, cv=5,
                      return_train_score=True)
    dt.fit(x_train_dr_svd, y_train_dr_svd.values.ravel())
    y_pred = dt.predict(x_test_dr_svd)

    mse_dt = MSE(y_test_dr_svd, y_pred)
    rmse_dt = mse_dt ** (1 / 2)

    print('mse:', mse_dt)
    print('rmse:', rmse_dt)
    print('best score:', dt.best_score_)
    print('best param:', dt.best_params_)

    df = pd.DataFrame(dt.cv_results_)
    print(df)

    pd.DataFrame(dt.cv_results_).to_pickle('./decision_tree_model_svd_reduced.pkl')

def make_and_save_decision_tree_model_with_dimred_svd():
        dt = GridSearchCV(DecisionTreeRegressor(),
                          {'max_depth': [4, 5, 6, 7],
                           'min_samples_leaf': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]}, cv=5,
                          return_train_score=True)
        dt.fit(x_train_dr_svd, y_train_dr_svd.values.ravel())
        y_pred = dt.predict(x_test_dr_svd)

        mse_dt = MSE(y_test_dr_svd, y_pred)
        rmse_dt = mse_dt ** (1 / 2)

        print('mse:', mse_dt)
        print('rmse:', rmse_dt)
        print('best score:', dt.best_score_)
        print('best param:', dt.best_params_)

        df = pd.DataFrame(dt.cv_results_)
        print(df)

        pd.DataFrame(dt.cv_results_).to_pickle('./decision_tree_model_svd_reduced_higher.pkl')


def make_svr_model_with_dimred_svd():
    model = GridSearchCV(SVC(), {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1,2,3,4,5,6,7,8,9,10]}, cv=5, return_train_score=True)
    model.fit(x_train_dr_svd, y_train_dr_svd.values.ravel())

    y_pred = model.predict(x_test_dr_svd)
    mse_dt = MSE(y_test_dr_svd, y_pred)
    rmse_dt = mse_dt ** (1 / 2)

    print('rmse:', rmse_dt)
    print('best score:', model.best_score_)
    print('best param:', model.best_params_)

    model.score(x_test_dr_svd, y_test_dr_svd)

    pd.DataFrame(model.cv_results_).to_pickle('./svr_model_svd_reduced.pkl')

def teszt():
    load = pickle.load('./svr_model.pkl')
    df = pd.DataFrame(load)
    print(df.head(5))

teszt()

make_and_save_decision_tree_model()
make_and_save_decision_tree_model_with_dimred_pca()
make_and_save_decision_tree_model_with_dimred_svd()

make_svr_model()
make_svr_model_with_dimred_pca()
make_svr_model_with_dimred_svd()
