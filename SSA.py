import matplotlib.pyplot as plt
import numpy as np


# Singular Spectrum Analysis 奇異譜分析方法

class SSA:

    def __init__(self):
        self.sequence = None
        self.series = None
        self.U = None
        self.sigma = None
        self.VT = None
        self.A = None
        self.windowLen = None
        self.seriesLen = None
        self.K = None
        self.X = None

    def svd(self):
        self.U, self.sigma, self.VT = np.linalg.svd(self.X, full_matrices=False)
        for i in range(self.VT.shape[0]):
            self.VT[i, :] *= self.sigma[i]
        self.A = self.VT

    def regroup(self):
        self.sequence = np.zeros((self.windowLen, self.K))
        for i in range(self.windowLen):
            for j in range(self.windowLen - 1):
                for m in range(j + 1):
                    self.sequence[i, j] += self.A[i, j - m] * self.U[m, i]
                self.sequence[i, j] /= (j + 1)
            for j in range(self.windowLen - 1, self.K):
                for m in range(self.windowLen):
                    self.sequence[i, j] += self.A[i, j - m] * self.U[m, i]
                self.sequence[i, j] /= self.windowLen

    def count(self, windowLen, series):
        self.windowLen = windowLen
        self.seriesLen = len(series)
        self.K = self.seriesLen - windowLen + 1
        self.X = np.zeros((windowLen, self.K))
        self.series = series

        for i in range(self.K):
            self.X[:, i] = series[i:i + windowLen]

        self.svd()
        self.regroup()

    def output(self):
        return self.sequence


series_rand = np.random.randn(500)

windowLen = 20
seriesLen = len(series_rand)

series = np.zeros(seriesLen)
for i in range(1, seriesLen):
    series[i] = np.sum(series_rand[:i])

ssa = SSA()
ssa.count(windowLen, series)
sequence = ssa.output()

plt.figure()
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    ax.plot(sequence[i, :])

plt.figure(2)
plt.plot(series)
plt.figure(3)
plt.plot(np.sum(sequence[:3], axis=0))  # 前三重要的主成分序列相加
plt.show()
