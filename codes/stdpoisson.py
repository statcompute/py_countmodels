import numpy, scipy
from statsmodels.base.model import GenericLikelihoodModel as gll

def _ll_stdpoisson(y, x, beta):
  mu = numpy.exp(numpy.dot(x, beta))
  pr = numpy.exp(-mu) * numpy.power(mu, y) / scipy.special.factorial(y)
  ll = numpy.log(pr)
  return(ll)

def stdpoisson(Y, X):
  class stdpoisson(gll):
    def __init__(self, endog, exog, **kwds):
      super(stdpoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      beta = params
      ll = _ll_stdpoisson(self.endog, self.exog, beta)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, **kwds):
      if start_params == None:
        start_params = numpy.zeros(self.exog.shape[1])
      return(super(stdpoisson, self).fit(start_params = start_params,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  return(stdpoisson(_Y, _X))
