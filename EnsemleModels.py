import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
import matplotlib as mpl
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class train(object):

    def __init__(self):
        self.optimizers=['neg_mean_squared_error', 'r2', 'explained_variance']
        [self.x_train, self.x_test, self.y_train, self.y_test]=\
            [np.loadtxt('./files/x_train.csv', delimiter=",", dtype=float), np.loadtxt('./files/x_test.csv',delimiter=",", dtype=float),
             np.loadtxt('./files/y_train.csv',delimiter=",", dtype=float), np.loadtxt('./files/y_test.csv',delimiter=",", dtype=float)]
    def save_pickle_dataframe(self, path, dataframe):
        dataframe.to_pickle(path)

    #1. probalkozas
    def randomForest_train(self,postfix):
        for optimizer in self.optimizers:
            rfr = RandomForestRegressor()
            sp = SelectPercentile(mutual_info_regression)
            param_best_grid={
                'sp__percentile': np.linspace(10.0, 100.0, 10),
                'rfr__n_estimators': range(50, 550, 50)
            }
            cv = StratifiedKFold(n_splits=10, shuffle=True)
            cv = StratifiedKFold(n_splits=10, shuffle=True)
            pipe = Pipeline(steps=[('sp', sp), ('rfr', rfr)])
            gridsearch = GridSearchCV(pipe, param_best_grid,scoring=optimizer, cv=cv, verbose=10, n_jobs=10)
            gridsearch.fit(self.x_train, self.y_train)

            df=pd.DataFrame(gridsearch.cv_results_)
            self.save_pickle_dataframe('./random_forest_{0}_{1}.pkl'.format(optimizer,postfix),df )

    #3. probalkozas
    def randomForest_train_poly(self,postfix):
        for optimizer in self.optimizers:
            rfr = RandomForestRegressor()
            sp = SelectPercentile(mutual_info_regression)
            param_best_grid={
                'sp__percentile': np.linspace(10.0, 100.0, 20),
                'rfr__n_estimators': range(50, 550, 100)
            }
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            cv = StratifiedKFold(n_splits=10, shuffle=True)
            pipe = Pipeline(steps=[('sp', sp),('poly', poly), ('rfr', rfr)])
            gridsearch = GridSearchCV(pipe, param_best_grid,scoring=optimizer, cv=cv, verbose=10, n_jobs=10)
            gridsearch.fit(self.x_train, self.y_train)

            df=pd.DataFrame(gridsearch.cv_results_)
            self.save_pickle_dataframe('./random_forest_{0}_{1}.pkl'.format(optimizer, postfix),df )


    #gradientboosting 1 probalkozas
    def gradientboosting_train(self):
        for optimizer in self.optimizers:
            gbr = GradientBoostingRegressor()
            sp = SelectPercentile(mutual_info_regression)
            param_best_grid = {
                'sp__percentile': np.linspace(10.0, 100.0, 10),
                'gbr__loss': ['ls', 'lad', 'huber', 'quantile'],
                'gbr__learning_rate': [0.1, 0.01, 0.001]
            }
            cv = StratifiedKFold(n_splits=10, shuffle=True)
            pipe = Pipeline(steps=[('sp', sp), ('gbr', gbr)])
            gridsearch = GridSearchCV(pipe, param_best_grid, scoring=optimizer, cv=cv, verbose=10, n_jobs=10)
            gridsearch.fit(self.x_train, self.y_train)

            df = pd.DataFrame(gridsearch.cv_results_)
            self.save_pickle_dataframe('./gbr_{0}.pkl'.format(optimizer), df)

    #gradientboosting-abrageneralas: igazabol notebookban van benne.
    def examine_output_gbr(self, filename):
        df = pd.read_pickle(filename)
        best_of = df[df['rank_test_score'] == 1]
        print('osszesitett: ', best_of[['param_gbr__loss', 'param_sp__percentile','param_gbr__learning_rate', 'mean_test_score']])
        for loss in ['ls', 'lad', 'huber', 'quantile']:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            df2=df[df['param_gbr__loss']==loss]
            triang = mtri.Triangulation(df2['param_sp__percentile'],
                                        df2['param_gbr__learning_rate'])
            ax.plot_trisurf(triang, df2['mean_test_score'], cmap='Blues')
            fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
            fake2Dline2 = mpl.lines.Line2D([1], [1], linestyle="none", c='r', marker='o')
            ax.legend([fake2Dline, fake2Dline2], ['Validacios score', 'Train score'], numpoints=1)

            ax.scatter(df2['param_sp__percentile'], df2['param_gbr__learning_rate'],
                       df2['mean_test_score'], marker='.', s=10, c="black", alpha=0.5)
            ax.plot_trisurf(triang, df2['mean_train_score'], cmap='Reds')

            ax.scatter(df2['param_sp__percentile'], df2['param_gbr__learning_rate'],
                       df2['mean_train_score'], marker='.', s=10, c="black", alpha=0.5)
            ax.view_init(elev=60, azim=-45)
            if df[df['rank_test_score'] == 1]['param_gbr__loss'].values[0]==loss:
                ax.scatter(df[df['rank_test_score'] == 1]['param_sp__percentile'].values,
                           df[df['rank_test_score'] == 1]['param_gbr__learning_rate'].values,
                           df[df['rank_test_score'] == 1]['mean_test_score'].values, marker='.', s=100, c="red",
                           alpha=1.0)
                ax.scatter(df[df['rank_test_score'] == 1]['param_sp__percentile'].values,
                           df[df['rank_test_score'] == 1]['param_gbr__learning_rate'].values,
                           df[df['rank_test_score'] == 1]['mean_train_score'].values, marker='.', s=100, c="red",
                           alpha=1.0)


            print(df['mean_test_score'])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
            print(df.columns)
            print(df)

    #randomforest eredmenyenek vizsgalata, ez is a notebookban van helyesen.
    def examine_output_randomforest(self, filename):
        df=pd.read_pickle(filename)
        best_of=df[df['rank_test_score'] == 1]
        print('osszesitett: ',best_of[['param_rfr__n_estimators', 'param_sp__percentile', 'mean_test_score']])
        fig=plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        triang=mtri.Triangulation(df['param_rfr__n_estimators'],
                                  df['param_sp__percentile'])
        ax.plot_trisurf(triang, df['mean_test_score'], cmap='jet')

        ax.plot_trisurf(triang, df['mean_train_score'], cmap='jet')
        print('param: ', df[df['rank_test_score'] == 1]['param_rfr__n_estimators'].values[0])

        ax.scatter(df[df['rank_test_score'] == 1]['param_rfr__n_estimators'].values, df[df['rank_test_score'] == 1]['param_sp__percentile'].values,
                   df[df['rank_test_score'] == 1]['mean_test_score'].values, marker='.', s=100, c="red", alpha=1.0)
        print('best_of: ',df['param_rfr__n_estimators'][df['rank_test_score'] == 1])
        print('itt: ',best_of.columns)
        ax.view_init(elev=60, azim=-45)
        print(df['mean_test_score'])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        print(df.columns)
        print(df)

    #3. probalkozas
    def random_tree_with_standard_data_train_poly(self, postfix):
        scaler=StandardScaler()
        x_train=self.x_train
        x_test=self.x_test
        scaler.fit_transform(self.x_train)
        scaler.transform(self.x_test)
        self.randomForest_train_poly(postfix)
        self.x_train=x_train
        self.x_test=x_test

    #2. probalkozas
    def random_tree_with_standard_data_train(self, postfix):
        scaler = StandardScaler()
        x_train = self.x_train
        x_test = self.x_test
        scaler.fit_transform(self.x_train)
        scaler.transform(self.x_test)
        self.randomForest_train(postfix)
        self.x_train = x_train
        self.x_test = x_test

    #2.-3. probalkozas tesztelese
    def random_tree_with_standard_data_test(self, filepath):
        scaler = StandardScaler()
        x_train = self.x_train
        x_test = self.x_test
        scaler.fit_transform(self.x_train)
        scaler.transform(self.x_test)

        self.examine_output_randomforest(filepath)
        self.x_train = x_train
        self.x_test = x_test

    #random forest tesztelese, igazabol felhasznalva lsd: osszesitett notebook
    def test_random_forest(self, filepath, score):
        df=pd.read_pickle(filepath)
        best_of=df[df['rank_test_score'] == 1]

        rfr = RandomForestRegressor(n_estimators=best_of['param_rfr__n_estimators'].vallues[0])
        sp = SelectPercentile(mutual_info_regression, percentile=best_of['param_sp__percentile'].values[0])
        model= Pipeline(steps=[('sp', sp), ('rfr', rfr)])

        model.fit(self.x_train, self.y_train)
        y_pred=model.predict(self.x_test)
        if score=='r2':
            return r2_score(self.y_test, y_pred)
        else:
            return mean_squared_error(self.y_test, y_pred)


t=train()
#1. proba
t.randomForest_train('10')
#2. proba
t.random_tree_with_standard_data_train('standard')
#3. proba
t.random_tree_with_standard_data_train('poly')

#t.random_tree_with_standard_data_test('random_forest_neg_mean_squared_error_standard.pkl')
#t.random_tree_with_standard_data_test('random_forest_r2.pkl')
#t.examine_output_randomforest('random_forest_neg_mean_squared_error.pkl')
#t.examine_output_randomforest('random_forest_neg_mean_squared_error_standard.pkl')
#t.examine_output_randomforest('random_forest_explained_variance.pkl')
#t.examine_output_randomforest('random_forest_neg_mean_squared_error_10.pkl')
#t.examine_output_randomforest('random_forest_neg_mean_squared_error_3.pkl')
#t.examine_output_gbr('gbr_neg_mean_squared_error.pkl')
#t.examine_output_randomforest('random_forest_r2_10.pkl')
#t.examine_output_randomforest('random_forest_r2_3.pkl')
t.examine_output_gbr('gbr_neg_mean_squared_error.pkl')
#t.examine_output_randomforest('random_forest_explained_variance_10.pkl')
#t.examine_output_randomforest('random_forest_explained_variance_3.pkl')
#t.examine_output_gbr('gbr_explained_variance_10.pkl')
