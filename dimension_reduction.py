import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD

df = pd.read_csv('./summed_data_to_train.csv')
df = df.drop(['Unnamed: 0'], axis=1)
y = df['price']
x = df
x = x.drop(['price'], axis=1)
x = x.values

def save_table(array, path):
    np.savetxt(path, array, delimiter=",", fmt='%lf')

#pca
def pca_reduction():
    pca = PCA(n_components=12)
    x = pca.fit(df).transform(df)
    print(pca.explained_variance_)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

    save_table(x_train, './files/x_train_pca_reduced.csv')
    save_table(y_train, './files/y_train_pca_reduced.csv')
    save_table(x_test, './files/x_test_pca_reduced.csv')
    save_table(y_test, './files/y_test_pca_reduced.csv')

#singular value decomposition
def svd_reduction():
    svd = TruncatedSVD(n_components=12)
    x = svd.fit(df).transform(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

    save_table(x_train, './files/x_train_svd_reduced.csv')
    save_table(y_train, './files/y_train_svd_reduced.csv')
    save_table(x_test, './files/x_test_svd_reduced.csv')
    save_table(y_test, './files/y_test_svd_reduced.csv')

pca_reduction()
svd_reduction()






