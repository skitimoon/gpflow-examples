from gpflow.models import SVGP
from gpflow.decors import autoflow
from gpflow import settings


class SVGPCoregion(SVGP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @autoflow((settings.float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var, Xnew)
