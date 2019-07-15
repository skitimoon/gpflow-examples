import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt

print(gpflow.__version__)
N = 100
X = np.random.uniform(-2 * np.pi, 2 * np.pi, (N, 1))
Y = np.sin(X) + 0.2 * np.random.randn(N, 1)
# print(X)
# print(Y)
# plt.scatter(X, Y)
# plt.show()

k = gpflow.kernels.SquaredExponential(1)
from gaussian_likelihood import Gaussian
m = gpflow.models.VGP(X, Y, k, Gaussian())
print(m)
Xpred = np.linspace(-3 * np.pi, 3 * np.pi, 200)[:, None]

print(m.compute_log_likelihood())
mean, var = m.predict_f(Xpred)
plt.scatter(X, Y)
plt.plot(Xpred, mean)
plt.fill_between(Xpred.flat,
                 mean.flat + 2 * np.sqrt(var.flat),
                 mean.flat - 2 * np.sqrt(var.flat),
                 alpha=0.4)
# plt.show()

mean, var = m.predict_y(Xpred)
plt.scatter(X, Y)
plt.plot(Xpred, mean)
plt.fill_between(Xpred.flat,
                 mean.flat + 2 * np.sqrt(var.flat),
                 mean.flat - 2 * np.sqrt(var.flat),
                 alpha=0.2)
plt.show()

opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)
print(m)
print(m.compute_log_likelihood())

mean, var = m.predict_f(Xpred)
plt.scatter(X, Y)
plt.plot(Xpred, mean)
plt.fill_between(Xpred.flat,
                 mean.flat + 2 * np.sqrt(var.flat),
                 mean.flat - 2 * np.sqrt(var.flat),
                 alpha=0.2)
plt.show()

mean, var = m.predict_y(Xpred)
plt.scatter(X, Y)
plt.plot(Xpred, mean)
plt.fill_between(Xpred.flat,
                 mean.flat + 2 * np.sqrt(var.flat),
                 mean.flat - 2 * np.sqrt(var.flat),
                 alpha=0.2)
plt.show()
