import numpy, scipy
from statsmodels.base.model import GenericLikelihoodModel

def _negll_poisson(y, X, beta):
  mu = numpy.exp(numpy.dot(X, beta))
  pr = numpy.exp(-mu) * numpy.power(mu, y) / scipy.special.factorial(y)
  ll = numpy.log(pr)
  return(-ll)
  
class MyPoisson(GenericLikelihoodModel):
  def __init__(self, endog, exog, **kwds):
    super(MyPoisson, self).__init__(endog, exog, **kwds)

  def nloglikeobs(self, params):
    beta = params
    ll = _negll_poisson(self.endog, self.exog, beta)
    return(ll)

  def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, **kwds):
    start_params = numpy.zeros(self.exog.shape[1])
    start_params[-1] = numpy.log(self.endog.mean())
    return(super(MyPoisson, self).fit(start_params = start_params,
                                      maxiter = maxiter, maxfun = maxfun, **kwds))

import statsmodels.api as sm

df = sm.datasets.get_rdataset("medpar", "COUNT", cache = True).data

Y = df.los

X = df.loc[:, ["type2", "type3", "hmo", "white"]]

X["constant"] = 1

mod = MyPoisson(Y, X)
res = mod.fit()
res.summary()
