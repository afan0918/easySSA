import matplotlib.pyplot as plt
import numpy as np


class SSA:
    def svd(self):
        # Perform Singular Value Decomposition on the trajectory matrix X
        self.U, self.sigma, self.VT = np.linalg.svd(self.X, full_matrices=False)

        # Scale the right singular vectors (VT) by singular values (sigma)
        for i in range(self.VT.shape[0]):
            self.VT[i, :] *= self.sigma[i]
        self.A = self.VT

    def regroup(self):
        # Reconstruct the time series using the components obtained from SVD
        self.sequence = np.zeros((self.windowLen, self.K))
        for i in range(self.windowLen):
            for j in range(self.windowLen - 1):
                for m in range(j + 1):
                    # Reconstruct each element of the sequence
                    self.sequence[i, j] += self.A[i, j - m] * self.U[m, i]
                self.sequence[i, j] /= (j + 1)
            for j in range(self.windowLen - 1, self.K):
                for m in range(self.windowLen):
                    self.sequence[i, j] += self.A[i, j - m] * self.U[m, i]
                self.sequence[i, j] /= self.windowLen

    def count(self, windowLen, series):
        # Set the window length and compute the trajectory matrix X
        self.windowLen = windowLen
        self.seriesLen = len(series)
        self.K = self.seriesLen - windowLen + 1
        self.X = np.zeros((windowLen, self.K))
        self.series = series

        for i in range(self.K):
            # Construct the trajectory matrix X from the time series
            self.X[:, i] = series[i:i + windowLen]

        # Perform Singular Spectrum Analysis
        self.svd()
        self.regroup()

    def output(self):
        # Return the reconstructed sequence
        return self.sequence


if __name__ == '__main__':
    series_rand = np.random.randn(500)

    windowLen = 10
    seriesLen = len(series_rand)

    series = np.zeros(seriesLen)
    for i in range(1, seriesLen):
        series[i] = np.sum(series_rand[:i])

    # Initialize SSA object
    ssa = SSA()

    # Perform Singular Spectrum Analysis
    ssa.count(windowLen, series)

    # Get the reconstructed sequence
    sequence = ssa.output()

    # Plot results
    plt.figure()
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        ax.plot(sequence[i, :])

    plt.figure(2)
    plt.plot(series)
    plt.figure(3)
    plt.plot(np.sum(sequence[:3], axis=0))  # Sum of the first three important principal components
    plt.show()
