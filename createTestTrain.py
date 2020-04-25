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

def save_table(array, path):
    np.savetxt(path, array, delimiter=",", fmt='%lf')

save_table(x_train, './files/x_train.csv')
save_table(y_train, './files/y_train.csv')
save_table(x_test, './files/x_test.csv')
save_table(y_test, './files/y_test.csv')