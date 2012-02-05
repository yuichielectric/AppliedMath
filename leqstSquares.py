import numpy as np
import matplotlib.pyplot as plt

class Figure(object):
    def __init__(self):
        self.figure_num = 0

    def _get_figure_num(self):
        self.figure_num += 1
        return self.figure_num

    def _get_approximate_f(self, x, y, rank):
        matrix = []
        for i in range(rank, 0, -1):
            matrix.append(x ** i)
        matrix.append(np.ones(len(x)))
        A = np.vstack(matrix).T
        coeffs = np.linalg.lstsq(A, y)[0]
        return np.poly1d(coeffs)

    def fit(self, xs, ys, rank = 1):
        x = np.array(xs)
        y = np.array(ys)
        f = self._get_approximate_f(x, y, rank)
        plt.figure(self._get_figure_num())
        plt.plot(x, y, 'o', label='Original data', markersize=8)
        fx = np.arange(min(x), max(x), 0.02)
        fy = [f(x) for x in fx]
        plt.plot(fx, fy, 'r', label='Fitted line')
        plt.legend()

    def show(self):
        plt.show()

# Sample code
if __name__ == '__main__':
    figure = Figure()

    # ex.1.2 (P. 4)
    xs = [4, 15, 30, 100]
    ys = [-17, -4, -7, 50]
    figure.fit(xs, ys)

    # ex.1.3 (P.6)
    xs = [5.6, 5.8, 6.0, 6.2, 6.4, 6.4, 6.4, 6.6, 6.8]
    ys = [30, 26, 33, 31, 33, 35, 37, 36, 33]
    figure.fit(xs, ys)

    # ex.1.5 (P.10)
    xs = [-1, 0, 0, 1]
    ys = [0, -2, -1, 1]
    figure.fit(xs, ys)

    # original sample
    xs = np.linspace(0, 10, 200)
    ys = np.cos(xs) + np.random.normal(0, 0.2, 200)
    figure.fit(xs, ys, 5)

    figure.show()
