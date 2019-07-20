import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import numpy as np
import matplotlib.pyplot as plt
tfd = tfp.distributions

np.random.seed(123)

X1 = np.sort(np.random.rand(600))[:, None]
X2 = np.sort(np.random.rand(500))[:, None]
X = [X1, X2]
print('X1', X1.shape)
print('X2', X2.shape)

def experiment_true_u_functions(X_list):
    u_functions = []
    for X in X_list:
        u_task = np.empty((X.shape[0],2))
        u_task[:,0,None] = 4.5*np.cos(2*np.pi*X + 1.5*np.pi) - \
                           3*np.sin(4.3*np.pi*X + 0.3*np.pi) + \
                           5*np.cos(7*np.pi * X + 2.4*np.pi)
        u_task[:,1,None] = 4.5*np.cos(1.5*np.pi*X + 0.5*np.pi) + \
                   5*np.sin(3*np.pi*X + 1.5*np.pi) - \
                   5.5*np.cos(8*np.pi * X + 0.25*np.pi)

        u_functions.append(u_task)
    return u_functions

def W_lincombination():
    W_list = []
    # q=1
    Wq1 = np.array(([[-0.5],[0.1]]))
    W_list.append(Wq1)
    # q=2
    Wq2 = np.array(([[-0.1],[.6]]))
    W_list.append(Wq2)
    return W_list

def experiment_true_f_functions(true_u, X_list):
    true_f = []
    W = W_lincombination()
    # D=1
    for d in range(2):
        f_d = np.zeros((X_list[d].shape[0], 1))
        for q in range(2):
            f_d += W[q][d].T*true_u[d][:,q,None]
        true_f.append(f_d)
    return true_f

trueU = experiment_true_u_functions(X)
trueF = experiment_true_f_functions(trueU, X)
plt.plot(X1, trueF[0])
plt.plot(X2, trueF[1])
plt.savefig('F.png')
plt.clf()

norm = tfd.Normal(trueF[0], 1)
bern = tfd.Bernoulli(probs=tf.sigmoid(trueF[1]))
Y1 = norm.sample().numpy()
Y2 = bern.sample().numpy()
print(Y1[:4])
print(Y2[:4])
plt.clf()

# plt.figure(figsize=(8, 4))
plt.plot(X1, Y1, 'x')
plt.plot(X2, Y2, 'x')
# plt.savefig('Y.png')
plt.plot(X1, trueF[0])
plt.plot(X2, trueF[1])
plt.savefig('FY.png')
plt.clf()

# Augment the input with ones or zeros to indicate the required output dimension
X1_augmented = np.hstack((X1, np.zeros_like(X1)))
X2_augmented = np.hstack((X2, np.ones_like(X2)))
X_augmented = np.vstack((X1_augmented, X2_augmented))

# Augment the Y data
Y1_augmented = np.hstack((Y1, np.zeros_like(Y1)))
Y2_augmented = np.hstack((Y2, np.ones_like(Y2)))
Y_augmented = np.vstack((Y1_augmented, Y2_augmented))

print('X_augmented', X_augmented.shape)
print('Y_augmented', Y_augmented.shape)
print(X_augmented[:4])
print(Y_augmented[:4])

# Generating missing data (gap)
X2test = X2[np.r_[351:450],:]
Y2test = Y2[np.r_[351:450],:]

# X2train_s1 = X[1][1:351,:]
# X2train_s2 = X[1][450:,:]

X2train = np.delete(X2, np.s_[np.r_[351:450]], 0)
Y2train = np.delete(Y2, np.s_[np.r_[351:450]], 0)

plt.plot(X1, Y1, 'x')
plt.plot(X2train, Y2train, 'x')
plt.plot(X2test, Y2test, 'xr')
plt.savefig('Ytrain.png')

output_dim = 2  # Number of outputs
rank = 1  # Rank of W
Q = 1
k1 = gpflow.kernels.Matern32(1, active_dims=[0])  # Base kernel
coreg = gpflow.kernels.Coregion(output_dim=output_dim,
                                rank=rank,
                                active_dims=[1])  # Coregion kernel
coreg.W = np.random.rand(output_dim, rank)  # Initialise the W matrix
kern = k1 * coreg

for _ in range(Q-1):
    k1 = gpflow.kernels.Matern32(1, active_dims=[0])  # Base kernel
    coreg = gpflow.kernels.Coregion(1,
                                    output_dim=output_dim,
                                    rank=rank,
                                    active_dims=[1])  # Coregion kernel
    coreg.W = np.random.rand(output_dim, rank)  # Initialise the W matrix
    kern += k1 * coreg

# This likelihood switches between Gaussian noise with different variances for each f_i:
lik = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian(),
     gpflow.likelihoods.Bernoulli()])

# M = 5
# initialisation of inducing input locations (M random points from the training inputs)
# Z = np.vstack((X1_augmented[np.random.choice(len(X1), M)],
#                X2_augmented[np.random.choice(len(X2), M)]))
# create multioutput features from Z
# feature = mf.MixedKernelSharedMof(gpflow.features.InducingPoints(Z))

X_augmented = tf.cast(X_augmented, tf.float32)
Y_augmented = tf.cast(Y_augmented, tf.float32)
# now build the GP model as normal
m = gpflow.models.VGP(X_augmented, Y_augmented, kernel=kern, likelihood=lik, num_latent=1)
# m = gpflow.models.SVGP(X_augmented,
#                        Y_augmented,
#                        Z=Z,
#                        kern=kern,
#                        likelihood=lik,
#                        num_latent=1)
# # Here we specify num_latent=1 to avoid getting two outputs when predicting as Y_augmented is 2-dimensional

print(m)
print(m.log_likelihood())
# # fit the covariance function parameters
# gpflow.train.ScipyOptimizer().minimize(m, maxiter=notebook_niter(1000))


# def plot_gp(x, mu, var, color, label):
#     plt.plot(x, mu, color=color, lw=2, label=label)
#     plt.fill_between(x[:, 0], (mu - 2 * np.sqrt(var))[:, 0],
#                      (mu + 2 * np.sqrt(var))[:, 0],
#                      color=color,
#                      alpha=0.4)


# def plot(m):
#     plt.figure(figsize=(8, 4))
#     xtest = np.linspace(0, 1, 100)[:, None]
#     line, = plt.plot(X1, Y1, 'x', mew=2)
#     mu, var = m.predict_y(np.hstack((xtest, np.zeros_like(xtest))))
#     # print(mu)
#     print(mu.shape)
#     plot_gp(xtest, mu, var, line.get_color(), 'Y1')

#     line, = plt.plot(X2, Y2, 'x', mew=2)
#     mu, var = m.predict_y(np.hstack((xtest, np.ones_like(xtest))))
#     plot_gp(xtest, mu, var, line.get_color(), 'Y2')

#     plt.legend()


# plot(m)
# plt.savefig('m.png')
# plt.show()

# B = coreg.W.value @ coreg.W.value.T + np.diag(coreg.kappa.value)
# print('B =\n', B)
# plt.imshow(B)
# plt.show()
