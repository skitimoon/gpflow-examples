import gpflow
import gpflow.multioutput.features as mf
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 6)
from gpflow.test_util import notebook_niter
np.random.seed(123)

# make a dataset with two outputs, correlated, heavy-tail noise. One has more noise than the other.
X1 = np.random.rand(100, 1)  # Observed locations for first output
X2 = np.random.rand(50, 1) * 0.5  # Observed locations for second output

Y1 = np.sin(6 * X1) + np.random.randn(*X1.shape) * 0.03
Y2 = np.sin(6 * X2 + 0.7) + np.random.randn(*X2.shape) * 0.1

plt.figure(figsize=(8, 4))
plt.plot(X1, Y1, 'x', mew=2)
plt.plot(X2, Y2, 'x', mew=2)
plt.show()

# Augment the input with ones or zeros to indicate the required output dimension
X1_augmented = np.hstack((X1, np.zeros_like(X1)))
X2_augmented = np.hstack((X2, np.ones_like(X2)))
X_augmented = np.vstack((X1_augmented, X2_augmented))

# Augment the Y data
Y1_augmented = np.hstack((Y1, np.zeros_like(Y1)))
Y2_augmented = np.hstack((Y2, np.ones_like(Y2)))
Y_augmented = np.vstack((Y1_augmented, Y2_augmented))

print(X_augmented.shape)
print(Y_augmented.shape)

output_dim = 2  # Number of outputs
rank = 1  # Rank of W
k1 = gpflow.kernels.Matern32(1, active_dims=[0])  # Base kernel
coreg = gpflow.kernels.Coregion(1,
                                output_dim=output_dim,
                                rank=rank,
                                active_dims=[1])  # Coregion kernel
coreg.W = np.random.rand(output_dim, rank)  # Initialise the W matrix
kern = k1 * coreg

# This likelihood switches between Gaussian noise with different variances for each f_i:
lik = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian(),
     gpflow.likelihoods.Gaussian()])

M = 5
# initialisation of inducing input locations (M random points from the training inputs)
Z = np.vstack((X1_augmented[np.random.choice(len(X1), M)],
               X2_augmented[np.random.choice(len(X2), M)]))
# # create multioutput features from Z
# feature = mf.MixedKernelSharedMof(gpflow.features.InducingPoints(Z))

# now build the GP model as normal
# m = gpflow.models.VGP(X_augmented, Y_augmented, kern=kern, likelihood=lik, num_latent=1)
m = gpflow.models.SVGP(X_augmented,
                       Y_augmented,
                       Z=Z,
                       kern=kern,
                       likelihood=lik,
                       num_latent=1)
# Here we specify num_latent=1 to avoid getting two outputs when predicting as Y_augmented is 2-dimensional

print(m)
# print(m.read_trainables())
# fit the covariance function parameters
gpflow.train.ScipyOptimizer().minimize(m, maxiter=notebook_niter(1000))


def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, color=color, lw=2, label=label)
    plt.fill_between(x[:, 0], (mu - 2 * np.sqrt(var))[:, 0],
                     (mu + 2 * np.sqrt(var))[:, 0],
                     color=color,
                     alpha=0.4)


def plot(m):
    plt.figure(figsize=(8, 4))
    xtest = np.linspace(0, 1, 100)[:, None]
    line, = plt.plot(X1, Y1, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.zeros_like(xtest))))
    # print(mu)
    print(mu.shape)
    plot_gp(xtest, mu, var, line.get_color(), 'Y1')

    line, = plt.plot(X2, Y2, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.ones_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color(), 'Y2')

    plt.legend()


plot(m)
plt.show()

B = coreg.W.value @ coreg.W.value.T + np.diag(coreg.kappa.value)
print('B =\n', B)
plt.imshow(B)
plt.show()
