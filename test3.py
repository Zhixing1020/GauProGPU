import numpy as np
import math
import time

from matplotlib import pyplot as plt
from numba import cuda

import warnings

warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

rng = np.random.RandomState(0)
X = rng.uniform(0, 1000, 4000)[:, np.newaxis]

y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

X_ = np.linspace(0, 1000, 100)[:, np.newaxis]


class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 1.0, "sigma_f": 1.0}
        self.optimize = optimize
        self.blockNum = 1024
        self.threadNum = 32

    def fit(self, X, y, l=None, sigma=None):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        if l is not None:
            self.params['l'] = l
        if sigma is not None:
            self.params['sigma_f'] = sigma
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        st = time.time()
        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        et = time.time()
        print('kernel time cost', et - st, 's')

        st = time.time()
        Kff_inv = np.linalg.inv(Kff + 1e-7 * np.eye(len(self.train_X)))  # (N, N)
        et = time.time()
        print('inv time cost', et - st, 's')

        st = time.time()
        mu = self.dproduct(self.dproduct(Kfy.T, Kff_inv), self.train_y)
        cov = Kyy - self.dproduct(self.dproduct(Kfy.T, Kff_inv), Kfy)
        et = time.time()
        print('dot time cost', et - st, 's')

        return mu, cov

    def dproduct(self, X1, X2):
        m = X1.shape[0]
        if X1.ndim == 1:
            n = 1
            X1 = X1[:, np.newaxis]
        else:
            n = X1.shape[1]
        if X2.shape[0] != n:
            print("the dimension of X1 and X2 is not aligned")
            return
        if X2.ndim == 1:
            l = 1
            X2 = X2[:,np.newaxis]
        else:
            l = X2.shape[1]

        result = np.zeros((m, l), dtype=float)
        cudaX1 = cuda.to_device(X1)
        cudaX2 = cuda.to_device(X2)
        cudaRes = cuda.to_device(result)

        # cudaX1 = gpuarray.to_gpu(X1)
        # cudaX2 = gpuarray.to_gpu(X2)
        # cudaRes = gpuarray.to_gpu(result)

        dotProduct[self.blockNum,self.threadNum](cudaX1,cudaX2,m,n,l,cudaRes)

        result = cudaRes.copy_to_host()
        #result = cudaRes.get()

        return result

    def kernel(self, X1, X2):
        dim = X1.shape[1]
        trailDim = X2.shape[1]
        if dim != trailDim:
            print("vectors in kernel have inequal dimension")
            return

        m = X1.shape[0]
        n = X2.shape[0]
        dist_matrix = np.zeros((m, n), dtype=float)
        result = np.zeros((m, n), dtype=float)

        cudaX1 = cuda.to_device(X1)
        cudaX2 = cuda.to_device(X2)
        cudaDM = cuda.to_device(dist_matrix)
        cudaRes = cuda.to_device(result)

        gaussian_kernel[self.blockNum,self.threadNum](cudaX1, cudaX2, m, n, dim, cudaDM, cudaRes, self.params['l'], self.params['sigma_f'])

        result = cudaRes.copy_to_host()
        # result = cudaRes.get()

        return result


@cuda.jit
def gaussian_kernel(x1, x2, m, n, dim, dist_matrix, result, l=1.0, sigma_f=1.0):
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x

    for i in range(bidx, m, cuda.gridDim.x):
        for j in range(n):
            for k in range(tidx, dim, cuda.blockDim.x):
                dist_matrix[i][j] += (x1[i][k] - x2[j][k]) ** 2
            result[i][j] = sigma_f ** 2 * math.exp(-0.5 * dist_matrix[i][j] / l ** 2)


@cuda.jit
def dotProduct(X1,X2,m,n,l,result):
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x

    for i in range(bidx,m,cuda.gridDim.x):
        for k in range(tidx,l,cuda.blockDim.x):
            for j in range(n):
                result[i][k] += X1[i][j]*X2[j][k]

def main():
    plt.figure()

    # GPU version
    # time_start = time.time()
    #
    # gpr = GPR()
    # gpr.fit(X, y)
    # y_mean, y_cov = gpr.predict(X_)
    #
    # time_end = time.time()
    # print('time cost', time_end - time_start, 's')

    # CPU version
    time_start = time.time()
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(X, y)
    y_mean, y_cov = gp.predict(X_, return_cov=True)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    # plot
    # plt.plot(X_[:,0], y_mean[:,0], 'k', lw=3, zorder=9)
    # plt.fill_between(X_[:,0], y_mean[:,0] - np.sqrt(np.diag(y_cov)),
    #                  y_mean[:,0] + np.sqrt(np.diag(y_cov)),
    #                  alpha=0.5, color='k')
    # plt.plot(X_[:,0], 0.5 * np.sin(3 * X_), 'r', lw=3, zorder=9)
    # plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    #
    # plt.show()


if __name__ == "__main__":
    main()