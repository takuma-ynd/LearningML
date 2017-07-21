import pandas as pd
from functions import *

# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import StandardScaler
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

# ワインデータをダウンロードする
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

num_classes = 13


# データの取り出し
# X(features): [[14.23, 1.71, 2.43, ...], ...]
# y(labels): [1, 1, ...]
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 訓練データ70%, テストデータ30%に分離する
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 標準化
sc = StandardScaler()
X_std = sc.fit_transform(X)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


print("Showing reduction by kernel PCA...")
X_kpca3 = rbf_kernel_pca(X_std, gamma=2.3, n_components=3)
wine_plot3d(X_kpca3, y)

print("Showing reduction by LLE")
from sklearn.manifold import LocallyLinearEmbedding
LLE3 = LocallyLinearEmbedding(n_components=3)
X_lle3 = LLE3.fit_transform(X_std, y)
wine_plot3d(X_lle3, y)

print("Showing reduction by MDS")
from sklearn.manifold import MDS
MDS3 = MDS(n_components=3)
X_mds3 = MDS3.fit_transform(X_std, y)
wine_plot3d(X_mds3, y)

print("Showing reduction by Isomap")
from sklearn.manifold import Isomap
ism3 = Isomap(n_components=3)
X_ism3 = ism3.fit_transform(X_std, y)
wine_plot3d(X_ism3, y)


