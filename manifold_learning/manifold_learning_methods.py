import os
import random
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
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import Isomap

from sklearn.linear_model import LogisticRegression

if "DISPLAY" not in os.environ:
    raise RuntimeError("empty enviromental variable:$DISPLAY\nSet up X11 Forwarding to show a graph.")

# ワインデータをダウンロードする
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

num_features = 13


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


# Logistic Regressionによる訓練および評価
kpca_list = [KernelPCA(n_components=i+1, kernel="rbf") for i in range(num_features-1)]
lle_list = [LocallyLinearEmbedding(n_components=i+1) for i in range(num_features-1)]
mds_list = [MDS(n_components=i+1) for i in range(num_features-1)]
ism_list = [Isomap(n_components=i+1) for i in range(num_features-1)]
sample_dimensions = [i for i in range(num_features)]
random.shuffle(sample_dimensions)

# 1次元〜(num_features-1)次元まで削減した各々のデータを格納
X_kpca_train = [kpca_list[i].fit_transform(X_train_std) for i in range(num_features - 1)]
X_kpca_test = [kpca_list[i].transform(X_test_std) for i in range(num_features -1)]

X_lle_train = [lle_list[i].fit_transform(X_train_std) for i in range(num_features - 1)]
X_lle_test = [lle_list[i].transform(X_test_std) for i in range(num_features -1)]

X_mds_train = [mds_list[i].fit_transform(X_train_std) for i in range(num_features - 1)]

X_ism_train = [ism_list[i].fit_transform(X_train_std) for i in range(num_features - 1)]
X_ism_test = [ism_list[i].transform(X_test_std) for i in range(num_features -1)]

X_rand_train = [X_train_std[:, sample_dimensions[:i+1]] for i in range(num_features -1)]
X_rand_test = [X_test_std[:, sample_dimensions[:i+1]] for i in range(num_features - 1)]

kpca_train_score, kpca_test_score = [], []
lle_train_score, lle_test_score = [], []
mds_train_score, mds_test_score = [], []
ism_train_score, ism_test_score = [], []
rand_train_score, rand_test_score = [], []

lr = LogisticRegression()
for i in range(num_features-1):
    kpca_lr = lr.fit(X_kpca_train[i], y_train)
    kpca_train_score.append(kpca_lr.score(X_kpca_train[i], y_train))
    kpca_test_score.append(kpca_lr.score(X_kpca_test[i], y_test))

    lle_lr = lr.fit(X_lle_train[i], y_train)
    lle_train_score.append(lle_lr.score(X_lle_train[i], y_train))
    lle_test_score.append(lle_lr.score(X_lle_test[i], y_test))

    mds_lr = lr.fit(X_mds_train[i], y_train)
    mds_train_score.append(mds_lr.score(X_mds_train[i], y_train))

    ism_lr = lr.fit(X_ism_train[i], y_train)
    ism_train_score.append(ism_lr.score(X_ism_train[i], y_train))
    ism_test_score.append(ism_lr.score(X_ism_test[i], y_test))

    rand_lr = lr.fit(X_rand_train[i], y_train)
    rand_train_score.append(rand_lr.score(X_rand_train[i], y_train))
    rand_test_score.append(rand_lr.score(X_rand_test[i], y_test))

# 比較グラフの描画
dimensions = [i+1 for i in range(num_features-1)]

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(18,8))

# 左の図についてグラフ描画
plt_kpca_l = axL.plot(dimensions, kpca_test_score, color="C0", marker="o")
plt_lle_l = axL.plot(dimensions, lle_test_score, color="C1", marker="o")
plt_ism_l = axL.plot(dimensions, ism_test_score, color="C2", marker="o")
plt_rand_l = axL.plot(dimensions, rand_test_score, color="C4", marker="s")
axL.grid(True)
axL.set_xlabel("dimension")
axL.set_ylabel("accuracy")
axL.set_title("Test accuracy comparison")
axL.legend((plt_kpca_l[0], plt_lle_l[0], plt_ism_l[0], plt_rand_l[0]), ("kernel-PCA", "LLE", "Isomap", "random"))

# 右の図についてグラフ描画
plt_kpca_r = axR.plot(dimensions, kpca_train_score, color="C0", marker="o")
plt_lle_r = axR.plot(dimensions, lle_train_score, color="C1", marker="o")
plt_ism_r = axR.plot(dimensions, ism_train_score, color="C2", marker="o")
plt_mds_r = axR.plot(dimensions, mds_train_score, color="C3", marker="o")
plt_rand_r = axR.plot(dimensions, rand_train_score, color="C4", marker="s")

axR.grid(True)
axR.set_xlabel("dimension")
axR.set_ylabel("accuracy")
axR.set_title("Training accuracy comparison")
axR.legend((plt_kpca_r[0], plt_lle_r[0], plt_ism_r[0], plt_mds_r[0], plt_rand_r[0]), ("kernel-PCA", "LLE", "Isomap", "MDS", "random"))

print("showing the result...")
plt.show()
