from scipy.spatial.distance import pdist, squareform
from scipy import exp
from numpy.linalg import eigh
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset   

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.linalg.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))

    return X_pc

def wine_plot3d(x, y):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[y == 1, 0], x[y == 1, 1], x[y == 1, 2], color='red', marker='o', alpha=0.5)
    ax.scatter(x[y == 2, 0], x[y == 2, 1], x[y == 2, 2], color='blue', marker='o', alpha=0.5)
    ax.scatter(x[y == 3, 0], x[y == 3, 1], x[y == 3, 2], color='green', marker='o', alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()
    
def wine_plot2d(x, y):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x[y==1, 0], x[y==1, 1], 
            color='red', marker='o', alpha=0.5)
    ax.scatter(x[y==2, 0], x[y==2, 1],
                color='blue', marker='o', alpha=0.5)
    ax.scatter(x[y==3, 0], x[y==3, 1],
                color='green', marker='o', alpha=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    plt.tight_layout()
    plt.show()
