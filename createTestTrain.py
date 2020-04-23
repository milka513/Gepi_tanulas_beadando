from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.pipeline import Pipeline


df = pd.read_csv('./summed_data_to_train.csv')
df = df.drop(['Unnamed: 0'], axis=1)
y = df['price']
x = df
x = x.drop(['price'], axis=1)
x = x.values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True
)

print('Start:')

cimkek = ['neg_mean_squared_error', 'r2']
for cimke in cimkek:
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    #baseline.fit(x_train, y_train)

    neighbor_params = [1,2,3,4,5,6,7,8,9,10,11,12]

    baseline = KNeighborsRegressor()
    sp = SelectPercentile(mutual_info_regression)

    param_best_grid={
        'sp__percentile': np.linspace(10.0, 100.0, 10),
        'baseline__n_estimators': range(50, 550, 50)
    }

    pipe = Pipeline(steps=[('sp', sp), ('baseline', baseline)])
    gridsearch = GridSearchCV(pipe, param_best_grid, scoring=cimke, cv=cv, verbose=10, n_jobs=6)
    gridsearch.fit(x_train, y_train)

    df = pd.DataFrame(gridsearch.cv_results_)

    print('train-acc: ',baseline.score(x_train,y_train))
    print('test-acc: ',baseline.score(x_test,y_test))

def save_table(array, path):
    np.savetxt(path, array, delimiter=",", fmt='%lf')

save_table(x_train, './files/x_train.csv')
save_table(y_train, './files/y_train.csv')
save_table(x_test, './files/x_test.csv')
save_table(y_test, './files/y_test.csv')