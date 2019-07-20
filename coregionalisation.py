import gpflow
from likelihoods import SwitchedLikelihoodCoregion
from models import SVGPCoregion
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = (12, 6)
np.random.seed(123)

# make a dataset with two outputs, correlated, heavy-tail noise. One has more noise than the other.
X1 = np.random.rand(100, 1)  # Observed locations for first output
X2 = np.random.rand(50, 1) * 0.5  # Observed locations for second output
X3 = np.random.rand(50, 1) * 0.5  # Observed locations for second output
X = [X1, X2, X3]

Y1 = np.sin(6 * X1) + np.random.randn(*X1.shape) * 0.03
Y2 = np.sin(6 * X2 + 0.7) + np.random.randn(*X2.shape) * 0.1
Y3 = -np.sin(6 * X3 + 0.7) + np.random.randn(*X3.shape) * 0.1
Y = [Y1, Y2, Y3]

plt.figure(figsize=(8, 4))
plt.plot(X1, Y1, 'x', mew=2)
plt.plot(X2, Y2, 'x', mew=2)
plt.plot(X3, Y3, 'x', mew=2)
plt.savefig('Y.png')
# plt.clf()
# plt.show()

# Augment the input with ones or zeros to indicate the required output dimension
X1_augmented = np.hstack((X1, np.zeros_like(X1)))
X2_augmented = np.hstack((X2, np.ones_like(X2)))
X3_augmented = np.hstack((X3, np.ones_like(X3) + 1))
X_augmented = np.vstack((X1_augmented, X2_augmented, X3_augmented))

# Augment the Y data
Y1_augmented = np.hstack((Y1, np.zeros_like(Y1)))
Y2_augmented = np.hstack((Y2, np.ones_like(Y2)))
Y3_augmented = np.hstack((Y3, np.ones_like(Y3) + 1))
Y_augmented = np.vstack((Y1_augmented, Y2_augmented, Y3_augmented))

print(X_augmented.shape)
print(Y_augmented.shape)

# Kernel
output_dim = 3  # Number of outputs
rank = 1  # Rank of W
Q = 3
k1 = gpflow.kernels.RBF(1, active_dims=[0])  # Base kernel
coreg = gpflow.kernels.Coregion(1,
                                output_dim=output_dim,
                                rank=rank,
                                active_dims=[1])  # Coregion kernel
coreg.W = np.random.rand(output_dim, rank)  # Initialise the W matrix
kern = k1 * coreg

for _ in range(Q-1):
    k1 = gpflow.kernels.RBF(1, active_dims=[0])  # Base kernel
    coreg = gpflow.kernels.Coregion(1,
                                    output_dim=output_dim,
                                    rank=rank,
                                    active_dims=[1])  # Coregion kernel
    coreg.W = np.random.rand(output_dim, rank)  # Initialise the W matrix
    kern += k1 * coreg

# This likelihood switches between Gaussian noise with different variances for each f_i:
# lik = gpflow.likelihoods.SwitchedLikelihood([
lik = SwitchedLikelihoodCoregion([
    gpflow.likelihoods.Gaussian(),
    gpflow.likelihoods.Gaussian(),
    gpflow.likelihoods.Gaussian()
])

M = 10
# initialisation of inducing input locations (M random points from the training inputs)
Z = np.vstack((X1_augmented[np.random.choice(len(X1), M)],
               X2_augmented[np.random.choice(len(X2), M)],
               X3_augmented[np.random.choice(len(X3), M)]))
# # create multioutput features from Z
# feature = mf.MixedKernelSharedMof(gpflow.features.InducingPoints(Z))

# now build the GP model as normal
# m = gpflow.models.VGP(X_augmented, Y_augmented, kern=kern, likelihood=lik, num_latent=1)
# m = gpflow.models.SVGP(X_augmented,
#                        Y_augmented,
#                        Z=Z,
#                        kern=kern,
#                        likelihood=lik,
#                        num_latent=1)

m = SVGPCoregion(X_augmented,
                 Y_augmented,
                 Z=Z,
                 kern=kern,
                 likelihood=lik,
                 num_latent=1)

# Here we specify num_latent=1 to avoid getting two outputs when predicting as Y_augmented is 2-dimensional

print(m)
# print(m.read_trainables())
# fit the covariance function parameters
# gpflow.train.ScipyOptimizer().minimize(m, maxiter=500, disp=True)
gpflow.train.AdamOptimizer(0.01).minimize(m, maxiter=500)
# gpflow.train.AdadeltaOptimizer(0.01).minimize(m, maxiter=2000)
print(m.compute_log_likelihood())
print(m)


def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, color=color, lw=2, label=label)
    plt.fill_between(x, (mu - 2 * np.sqrt(var)), (mu + 2 * np.sqrt(var)),
                     color=color,
                     alpha=0.4)


def plot_f(m):
    plt.figure(figsize=(8, 4))
    Xtest = np.linspace(0, 1, 100)[:, None]
    for i, Xi, Yi in zip(range(output_dim), X, Y):
        line, = plt.plot(Xi, Yi, 'x', mew=2)
        mu, var = m.predict_f(np.hstack((Xtest, np.zeros_like(Xtest) + i)))
        print(mu[:4])
        print(mu.shape)
        plot_gp(Xtest.flat, mu.flat, var.flat, line.get_color(), f'Y{i}')
    Xorig = np.linspace(0, 1, 100)
    Y1 = np.sin(6 * Xorig)
    Y2 = np.sin(6 * Xorig + 0.7)
    Y3 = -np.sin(6 * Xorig + 0.7)
    # plt.plot(Xorig, Y1, '--')
    # plt.plot(Xorig, Y2, '--')
    # plt.plot(Xorig, Y3, '--')
    # plt.title(f'Q = {Q}, M = {M}')
    plt.legend()


def plot_y(m):
    plt.figure(figsize=(8, 4))
    Xtest = np.linspace(0, 1, 100)[:, None]
    for i, Xi, Yi in zip(range(output_dim), X, Y):
        line, = plt.plot(Xi, Yi, 'x', mew=2)
        mu, var = m.predict_y(np.hstack((Xtest, np.zeros_like(Xtest) + i)))
        # mu, var = mu[:, 0], var[:, i]
        print(mu[:4])
        print(mu.shape)
        plot_gp(Xtest.flat, mu.flat, var.flat, line.get_color(), f'Y{i}')
    plt.legend()


plot_f(m)
plt.savefig(f'FpredQ{Q}M{M}.png')
plt.clf()
plot_y(m)
plt.savefig(f'YpredQ{Q}M{M}.png')
plt.clf()
# plt.show()

# B = coreg.W.value @ coreg.W.value.T + np.diag(coreg.kappa.value)
# print('B =\n', B)
# plt.imshow(B)
# plt.savefig(f'BQ{Q}M{M}.png')
# plt.clf()
# # plt.show()

# Xtest = np.linspace(0, 1, 100)[:, None]
# Fpred, Fvar = m.predict_f(
#     np.vstack((np.hstack((Xtest, np.zeros_like(Xtest))),
#                np.hstack((Xtest, np.zeros_like(Xtest) + 1)),
#                np.hstack((Xtest, np.zeros_like(Xtest) + 2)))))
# print(Fpred.shape)
# print(Fvar.shape)
# print(Fpred[:2])
# print(Fvar[:2])

# X = np.linspace(0, 1, 100)
# Y1 = np.sin(6 * X)
# Y2 = np.sin(6 * X + 0.7)
# Y3 = -np.sin(6 * X + 0.7)
# plt.plot(X, Y1)
# plt.plot(X, Y2)
# plt.plot(X, Y3)
# plt.savefig('data.png')
