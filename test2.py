import numpy as np
import time
import math

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as dry
from pycuda.compiler import SourceModule
import skcuda.linalg as sklin

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


rng = np.random.RandomState(0)
X = rng.uniform(0, 10, 1000)[:, np.newaxis]

y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

X_ = np.linspace(0, 10, 100)[:, np.newaxis]

class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 1.0, "sigma_f": 1.0}
        self.optimize = optimize
        self.blockNum = 1024
        self.threadNum = 32
        sklin.init()

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
        Kff = self.kernel2(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel2(X, X)  # (k, k)
        Kfy = self.kernel2(self.train_X, X)  # (N, k)
        et = time.time()
        print('kernel time cost', et - st, 's')

        st = time.time()
        Kff_gpu = gpuarray.to_gpu((Kff + 1e-7 * np.eye(len(self.train_X))).astype(np.float32))
        Kff_inv_gpu = sklin.inv(Kff_gpu)
        Kff_inv = Kff_inv_gpu.get()
        et = time.time()
        print('2inv time cost', et - st, 's')

        st = time.time()
        aT = np.random.randn(Kfy.shape[1], Kfy.shape[0])
        aT[:, :] = Kfy.T[:, :]
        KfyT_gpu = gpuarray.to_gpu((aT).astype(np.float32))
        Kfy_gpu = gpuarray.to_gpu(Kfy.astype(np.float32))
        trainy_gpu = gpuarray.to_gpu(self.train_y.astype(np.float32))
        Kff_inv_gpu = gpuarray.to_gpu(Kff_inv.astype(np.float32))
        gpu_t = sklin.dot(KfyT_gpu, Kff_inv_gpu)
        mu = sklin.dot(gpu_t, trainy_gpu).get()
        cov = Kyy - sklin.dot(gpu_t, Kfy_gpu).get()
        et = time.time()
        print('dot time cost', et - st, 's')
        return mu, cov

    def kernel2(self, X1, X2):
        dim = X1.shape[1]
        trailDim = X2.shape[1]
        if dim != trailDim:
            print("vectors in kernel have inequal dimension")
            return

        m = X1.shape[0]
        n = X2.shape[0]
        X1.astype(np.float32)
        X2.astype(np.float32)
        dist_matrix = np.zeros((m, n)).astype(np.float32)
        result = np.zeros((m, n)).astype(np.float32)

        mod = SourceModule(
                  """
                  #include<math.h>
                  
                  __global__ void gaussian_ken(float *x1, float *x2, float* dist_matrix,
                  float* result, float l, float sigma_f, int m, int n, int dim)
                  {
                    const int tidx = threadIdx.x;
                    const int bidx = blockIdx.x;
                    
                    for(int i=bidx; i<m; i+=gridDim.x)
                    {
                        for(int j=0;j<n; j++)
                        {
                            for(int k=tidx; k<dim; k+=blockDim.x)
                                dist_matrix[i*(n)+j] += pow((x1[i*(dim)+k] - x2[j*(dim)+k]),2);
                            result[i*(n)+j] = pow(sigma_f, 2) * exp(-0.5 * pow(dist_matrix[i*(n)+j] / (l), 2));
                        }
                    }
                  }
                  """
              )

        func = mod.get_function("gaussian_ken")
        func(dry.In(X1), dry.In(X2), dry.In(dist_matrix), dry.InOut(result), np.float32(self.params['l']), np.float32(self.params['sigma_f']), np.int32(m), np.int32(n), np.int32(dim),  block=(self.threadNum, 1,1),grid=(self.blockNum,1,1))

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

        for i in range(m):
            for j in range(n):
                for k in range(dim):
                    dist_matrix[i][j] += (X1[i][k] - X2[j][k]) ** 2
                result[i][j] = self.params['sigma_f'] ** 2 * math.exp(-0.5 * dist_matrix[i][j] / self.params['l'] ** 2)

        return result

def main():
    plt.figure()

    #GPU version
    time_start = time.time()

    gpr = GPR()
    gpr.fit(X, y)
    y_mean, y_cov = gpr.predict(X_)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    #CPU version
    # time_start = time.time()
    # kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    #          + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    # gp = GaussianProcessRegressor(kernel=kernel,
    #                               alpha=0.0).fit(X, y)
    # y_mean, y_cov = gp.predict(X_, return_cov=True)
    #
    # time_end = time.time()
    # print('time cost', time_end - time_start, 's')

    #plot
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