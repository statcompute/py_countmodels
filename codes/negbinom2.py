import numpy, scipy
from statsmodels.base.model import GenericLikelihoodModel as gll

def _ll_negbinom2(y, x, beta, alpha):
  mu = numpy.exp(numpy.dot(x, beta))
  a1 = 1 / alpha
  pr = scipy.special.gamma(y + a1) / (scipy.special.gamma(y + 1) * scipy.special.gamma(a1)) * \
       numpy.power(a1 / (a1 + mu), a1) * numpy.power(mu / (a1 + mu), y)
  ll = numpy.log(pr)
  return(ll)

def negbinom2(Y, X):
  class negbinom2(gll):
    def __init__(self, endog, exog, **kwds):
      super(negbinom2, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      alpha = params[-1]
      beta = params[:-1]
      ll = _ll_negbinom2(self.endog, self.exog, beta, alpha)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, **kwds):
      self.exog_names.append('_ALPHA')
      if start_params == None:
        start_params = numpy.append(numpy.zeros(self.exog.shape[1]), 0.5)
      return(super(negbinom2, self).fit(start_params = start_params,
                                        maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  return(negbinom2(_Y, _X))
