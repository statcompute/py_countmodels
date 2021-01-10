import numpy, scipy
from statsmodels.base.model import GenericLikelihoodModel

def _negll_poisson(y, X, beta):
  mu = numpy.exp(numpy.dot(X, beta))
  pr = numpy.exp(-mu) * numpy.power(mu, y) / scipy.special.factorial(y)
  ll = numpy.log(pr)
  return(-ll)
  
class StdPoisson(GenericLikelihoodModel):
  def __init__(self, endog, exog, **kwds):
    super(StdPoisson, self).__init__(endog, exog, **kwds)

  def nloglikeobs(self, params):
    beta = params
    ll = _negll_poisson(self.endog, self.exog, beta)
    return(ll)

  def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, **kwds):
    if start_params == None:
      start_params = numpy.zeros(self.exog.shape[1])
      start_params[-1] = numpy.log(self.endog.mean())
    return(super(StdPoisson, self).fit(start_params = start_params,
                                      maxiter = maxiter, maxfun = maxfun, **kwds))

import pandas

df = pandas.read_csv("data/credit_count.csv")

y = df.MAJORDRG

xnames = ['AGE', 'ACADMOS', 'MINORDRG', 'OWNRENT']

X = df.loc[:, xnames]

X["constant"] = 1

mdl = StdPoisson(y, X)

mdl.fit().summary()
