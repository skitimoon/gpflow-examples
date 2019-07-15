import tensorflow as tf
from gpflow.likelihoods import SwitchedLikelihood
from gpflow.decors import params_as_tensors_for

class SwitchedLikelihoodCoregion(SwitchedLikelihood):
    def __init__(self, likelihood_list, **kwargs):
        super().__init__(likelihood_list, **kwargs)

    def predict_mean_and_var(self, Fmu, Fvar, Xnew):
        ind = Xnew[:, -1]
        ind = tf.cast(ind, tf.int32)
        args = [Fmu, Fvar]
        args = zip(*[tf.dynamic_partition(X, ind, self.num_likelihoods) for X in args])

        with params_as_tensors_for(self, convert=False):
            funcs = [getattr(lik, 'predict_mean_and_var') for lik in self.likelihood_list]
        results = [f(*args_i) for f, args_i in zip(funcs, args)]

        mu = [result[0] for result in results]
        var = [result[1] for result in results]

        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_likelihoods)
        mu = tf.dynamic_stitch(partitions, mu)
        var = tf.dynamic_stitch(partitions, var)

        return mu, var
