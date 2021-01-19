# py_countreg/py_countreg.py
# exec(open('py_countreg/py_countreg.py').read())
# 0.0.1

import numpy, scipy
from statsmodels.base.model import GenericLikelihoodModel as gll
from statsmodels.api import Logit as logit


#################### 01. Standard Poisson Regression ####################


def _ll_stdpoisson(y, x, beta):
  """
  The function calculates the log likelihood function of a standard poisson 
  regression.
  Parameters:
    y    : the frequency outcome 
    x    : variables of the poisson regression
    beta : coefficients of the poisson regression 
  """

  mu = numpy.exp(numpy.dot(x, beta))
  pr = numpy.exp(-mu) * numpy.power(mu, y) / scipy.special.factorial(y)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def stdpoisson(Y, X):
  """
  The function estimates a standard poisson regression.
  Parameters:
    Y : a pandas series for the frequency outcome
    X : a pandas dataframe with model variables that are all numeric
  Example:
    stdpoisson(Y, X).fit().summary()   
  """

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


#################### 02. Negative Binomial Regression ####################


def _ll_negbinom2(y, x, beta, alpha):
  """
  The function calculates the log likelihood function of the negative binomial
  (NB-2) regression.
  Parameters:
    y     : the frequency outcome
    x     : variables of the negative binomial regression
    beta  : coefficients of the negative binomial regression
    alpha : the dispersion parameter of the negative binomial regression
  """

  mu = numpy.exp(numpy.dot(x, beta))
  a1 = 1 / alpha
  pr = scipy.special.gamma(y + a1) / (scipy.special.gamma(y + 1) * scipy.special.gamma(a1)) * \
       numpy.power(a1 / (a1 + mu), a1) * numpy.power(mu / (a1 + mu), y)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def negbinom2(Y, X):
  """
  The function estimates a negative binomial (NB-2) regression.
  Parameters:
    Y : a pandas series for the frequency outcome
    X : a pandas dataframe with model variables that are all numeric
  Example:
    negbinom2(Y, X).fit().summary()   
  """

  class negbinom2(gll):
    def __init__(self, endog, exog, **kwds):
      super(negbinom2, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      alpha = params[-1]
      beta = params[:-1]
      ll = _ll_negbinom2(self.endog, self.exog, beta, alpha)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_ALPHA')
      if start_params == None:
        start_params = numpy.append(numpy.zeros(self.exog.shape[1]), 1)
      return(super(negbinom2, self).fit(start_params = start_params, method = method,
                                        maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  return(negbinom2(_Y, _X))


#################### 03. Generalized Poisson Regression ####################


def _ll_genpoisson(y, x, beta, s):
  """
  The function calculates the log likelihood function of the generalized poisson 
  regression.
  Parameters:
    y    : the frequency outcome
    x    : variables of the generalized poisson regression
    beta : coefficients of the negative binomial regression
    s    : the scale parameter for the generalized poisson distribution
  """

  mu = numpy.exp(numpy.dot(x, beta))
  xi = numpy.exp(s)
  _a = mu * (1 - xi)
  pr = _a / scipy.special.factorial(y) * numpy.exp(-_a - xi * y) * \
       numpy.power(_a + xi * y, y - 1)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def genpoisson(Y, X):
  """
  The function estimates a generalized poisson regression.
  Parameters:
    Y : a pandas series for the frequency outcome
    X : a pandas dataframe with model variables that are all numeric
  Example:
    genpoisson(Y, X).fit().summary()   
  """
  class genpoisson(gll):
    def __init__(self, endog, exog, **kwds):
      super(genpoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      _s = params[-1]
      beta = params[:-1]
      ll = _ll_genpoisson(self.endog, self.exog, beta, _s)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_S')
      if start_params == None:
        start_params = numpy.append(p0, s0)
      return(super(genpoisson, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  m0 = stdpoisson(Y, X).fit(disp = 0)
  p0 = m0.params
  s0 = numpy.log(max(1e-4, 1 - numpy.power(numpy.mean(Y) / numpy.var(Y), 0.5)))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  return(genpoisson(_Y, _X))


#################### 04. Hurdle Poisson Regression ####################


def _ll_hdlpoisson(y, x1, x2, beta1, beta2):
  """
  The function calculates the log likelihood function of the hurdle poisson 
  regression.
  Parameters:
    y     : the frequency outcome
    x1    : variables for the probability model in the hurdle poisson regression
    x2    : variables for the count model in the hurdle poisson regression
    beta1 : coefficients for the probability model in the hurdle poisson regression
    beta2 : coefficients for the count model in the hurdle poisson regression
  """

  xb1 = numpy.dot(x1, beta1)
  xb2 = numpy.dot(x2, beta2)
  p0 = numpy.exp(xb1) / (1 + numpy.exp(xb1)) 
  mu = numpy.exp(xb2)
  i0 = numpy.where(y == 0, 1, 0)
  pr = p0 * i0 + \
       (1 - p0) * numpy.exp(-mu) * numpy.power(mu, y) / \
       ((1 - numpy.exp(-mu)) * scipy.special.factorial(y)) * (1 - i0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def hdlpoisson(Y, X1, X2):
  """
  The function estimates a hurdle poisson regression, which is the composite 
  between point mess at zero and a zero-trucated poisson distribution.
  Parameters:
    Y  : a pandas series for the frequency outcome
    X1 : a pandas dataframe with the probability model variables that are all numeric
    X2 : a pandas dataframe with the count model variables that are all numeric
  Example:
    hdlpoisson(Y, X1, X2).fit().summary()   
  """

  class hdlpoisson(gll):
    def __init__(self, endog, exog, **kwds):
      super(hdlpoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      d1 = _X1.shape[1]
      beta1 = params[:d1]
      beta2 = params[d1:]
      ll = _ll_hdlpoisson(self.endog, self.exog[:, :d1], self.exog[:, d1:], beta1, beta2)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      if start_params == None:
        start_params = numpy.concatenate([p10, p20])
      return(super(hdlpoisson, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X1 = X1.copy()
  _X2 = X2.copy()
  _X1.insert(loc = 0, column = "_CONST", value = 1)
  _X1.columns = ["P0:" + _ for _ in _X1.columns]
  m10 = logit(numpy.where(_Y == 0, 1, 0), _X1).fit(disp = 0)
  p10 = m10.params
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["MU:" + _ for _ in _X2.columns]
  m20 = stdpoisson(_Y[_Y > 0], X2[_Y > 0]).fit(disp = 0)
  p20 = m20.params
  _X = _X1.join(_X2)
  return(hdlpoisson(_Y, _X))


#################### 05. Zero-Inflated Poisson Regression ####################


def _ll_zifpoisson(y, x1, x2, beta1, beta2):
  """
  The function calculates the log likelihood function of the zero-inflated 
  poisson regression.
  Parameters:
    y     : the frequency outcome
    x1    : variables for the probability model in the zero-inflated poisson regression
    x2    : variables for the count model in the zero-inflated poisson regression
    beta1 : coefficients for the probability model in the zero-inflated poisson regression
    beta2 : coefficients for the count model in the zero-inflated poisson regression
  """

  xb1 = numpy.dot(x1, beta1)
  xb2 = numpy.dot(x2, beta2)
  p0 = numpy.exp(xb1) / (1 + numpy.exp(xb1)) 
  mu = numpy.exp(xb2)
  i0 = numpy.where(y == 0, 1, 0)
  pr = (p0 + (1 - p0) * numpy.exp(-mu)) * i0 + \
       (1 - p0) * numpy.exp(-mu) * numpy.power(mu, y) / \
       scipy.special.factorial(y) * (1 - i0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def zifpoisson(Y, X1, X2):
  """
  The function estimates a zero-inflated poisson regression, which is the 
  composite between point mess at zero and a standard poisson distribution.
  Parameters:
    Y  : a pandas series for the frequency outcome
    X1 : a pandas dataframe with the probability model variables that are all numeric
    X2 : a pandas dataframe with the count model variables that are all numeric
  Example:
    zifpoisson(Y, X1, X2).fit().summary()   
  """

  class zifpoisson(gll):
    def __init__(self, endog, exog, **kwds):
      super(zifpoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      d1 = _X1.shape[1]
      beta1 = params[:d1]
      beta2 = params[d1:]
      ll = _ll_zifpoisson(self.endog, self.exog[:, :d1], self.exog[:, d1:], beta1, beta2)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      if start_params == None:
        start_params = numpy.concatenate([p10, p20])
      return(super(zifpoisson, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X1 = X1.copy()
  _X2 = X2.copy()
  _X1.insert(loc = 0, column = "_CONST", value = 1)
  _X1.columns = ["P0:" + _ for _ in _X1.columns]
  m10 = logit(numpy.where(_Y == 0, 1, 0), _X1).fit(disp = 0)
  p10 = m10.params
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["MU:" + _ for _ in _X2.columns]
  m20 = stdpoisson(_Y[_Y > 0], X2[_Y > 0]).fit(disp = 0)
  p20 = m20.params
  _X = _X1.join(_X2)
  return(zifpoisson(_Y, _X))


#################### 06. Conway-Maxwell Poisson Regression ####################


def _ll_compoisson(y, x, beta, s):
  """
  The function calculates the log likelihood function of the Conway-Maxwell
  poisson regression.
  Parameters:
    y    : the frequency outcome
    x    : variables in the conway-maxwell poisson regression
    beta : coefficients in the conway maxwell poisson regression
    s    : the scale parameter in the Conway-Maxwell distribution and is equal to log(nv)
  """

  mu = numpy.exp(numpy.dot(x, beta))
  nv = numpy.exp(s) 
  _z = 0
  for _n in range(100):
    _z = _z + numpy.power(mu, _n) / numpy.power(scipy.special.factorial(_n), nv)

  pr = numpy.power(mu, y) / numpy.power(scipy.special.factorial(y), nv) * numpy.power(_z, -1)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def compoisson(Y, X):
  """
  The function estimates a Conway-Maxwell poisson regression.
  Parameters:
    Y : a pandas series for the frequency outcome
    X : a pandas dataframe with the probability model variables that are all numeric
  Example:
    compoisson(Y, X).fit().summary()   
  """

  class compoisson(gll):
    def __init__(self, endog, exog, **kwds):
      super(compoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      _s = params[-1]
      beta = params[:-1]
      ll = _ll_compoisson(self.endog, self.exog, beta, _s)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_S')
      if start_params == None:
        start_params = numpy.append(p0, s0)
      return(super(compoisson, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  m0 = stdpoisson(Y, X).fit(disp = 0)
  p0 = m0.params
  s0 = numpy.log(numpy.mean(Y) / numpy.var(Y))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  return(compoisson(_Y, _X))


#################### 07. Hurdle Negative Binomial Regression ####################


def _ll_hdlnegbin2(y, x1, x2, beta1, beta2, alpha):
  """
  The function calculates the log likelihood function of the hurdle negative 
  binomial regression.
  Parameters:
    y     : the frequency outcome
    x1    : variables for the probability model in the hurdle negative binomial regression
    x2    : variables for the count model in the hurdle negative binomial regression
    beta1 : coefficients for the probability model in the hurdle negative binomial regression
    beta2 : coefficients for the count model in the hurdle negative binomial regression
    alpha : the dispersion parameter in the negative binomial distribution 
  """

  xb1 = numpy.dot(x1, beta1)
  xb2 = numpy.dot(x2, beta2)
  p0 = numpy.exp(xb1) / (1 + numpy.exp(xb1)) 
  mu = numpy.exp(xb2)
  i0 = numpy.where(y == 0, 1, 0)
  a1 = 1 / alpha
  pr = p0 * i0 + \
       (1 - p0) / (1 - numpy.power(a1 / (a1 + mu), a1)) * \
       scipy.special.gamma(y + a1) / (scipy.special.gamma(y + 1) * scipy.special.gamma(a1)) * \
       numpy.power(a1 / (a1 + mu), a1) * numpy.power(mu / (a1 + mu), y) * (1 - i0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def hdlnegbin2(Y, X1, X2):
  """
  The function estimates a hurdle negative binomial regression, which is the 
  composite between point mess at zero and a zero-truncated negative binomial 
  distribution.
  Parameters:
    Y  : a pandas series for the frequency outcome
    X1 : a pandas dataframe with the probability model variables that are all numeric
    X2 : a pandas dataframe with the count model variables that are all numeric
  Example:
    hdlnegbin2(Y, X1, X2).fit().summary()   
  """

  class hdlnegbin2(gll):
    def __init__(self, endog, exog, **kwds):
      super(hdlnegbin2, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      d1 = _X1.shape[1]
      beta1 = params[:d1]
      beta2 = params[d1:-1]
      alpha = params[-1]
      ll = _ll_hdlnegbin2(self.endog, self.exog[:, :d1], self.exog[:, d1:], beta1, beta2, alpha)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_ALPHA')
      if start_params == None:
        start_params = numpy.concatenate([p10, p20])
      return(super(hdlnegbin2, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X1 = X1.copy()
  _X2 = X2.copy()
  _X1.insert(loc = 0, column = "_CONST", value = 1)
  _X1.columns = ["P0:" + _ for _ in _X1.columns]
  m10 = logit(numpy.where(_Y == 0, 1, 0), _X1).fit(disp = 0)
  p10 = m10.params
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["MU:" + _ for _ in _X2.columns]
  m20 = negbinom2(_Y[_Y > 0], X2[_Y > 0]).fit(disp = 0)
  p20 = m20.params
  _X = _X1.join(_X2)
  return(hdlnegbin2(_Y, _X))


#################### 08. Zero-Inflated Negative Binomial Regression ####################


def _ll_zifnegbin2(y, x1, x2, beta1, beta2, alpha):
  """
  The function calculates the log likelihood function of the zero-inflated 
  negative binomial regression.
  Parameters:
    y     : the frequency outcome
    x1    : variables for the probability model in the zero-inflated negative binomial regression
    x2    : variables for the count model in the zero-inflated negative binomial regression
    beta1 : coefficients for the probability model in the zero-inflated negative binomial regression
    beta2 : coefficients for the count model in the zero-inflated negative binomial regression
    alpha : the dispersion parameter in the negative binomial distribution 
  """

  xb1 = numpy.dot(x1, beta1)
  xb2 = numpy.dot(x2, beta2)
  p0 = numpy.exp(xb1) / (1 + numpy.exp(xb1)) 
  mu = numpy.exp(xb2)
  i0 = numpy.where(y == 0, 1, 0)
  a1 = 1 / alpha
  pr = (p0 + (1 - p0) * numpy.power(a1 / (a1 + mu), a1)) * i0 + \
       (1 - p0) * scipy.special.gamma(y + a1) / (scipy.special.gamma(y + 1) * scipy.special.gamma(a1)) * \
       numpy.power(a1 / (a1 + mu), a1) * numpy.power(mu / (a1 + mu), y) * (1 - i0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def zifnegbin2(Y, X1, X2):
  """
  The function estimates a zero-inflated negative binomial regression, which is the 
  composite between point mess at zero and a negative binomial distribution.
  Parameters:
    Y  : a pandas series for the frequency outcome
    X1 : a pandas dataframe with the probability model variables that are all numeric
    X2 : a pandas dataframe with the count model variables that are all numeric
  Example:
    zifnegbin2(Y, X1, X2).fit().summary()   
  """

  class zifnegbin2(gll):
    def __init__(self, endog, exog, **kwds):
      super(zifnegbin2, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      d1 = _X1.shape[1]
      beta1 = params[:d1]
      beta2 = params[d1:-1]
      alpha = params[-1]
      ll = _ll_zifnegbin2(self.endog, self.exog[:, :d1], self.exog[:, d1:], beta1, beta2, alpha)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_ALPHA')
      if start_params == None:
        start_params = numpy.concatenate([p10, p20])
      return(super(zifnegbin2, self).fit(start_params = start_params, method = method,
                                           maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X1 = X1.copy()
  _X2 = X2.copy()
  _X1.insert(loc = 0, column = "_CONST", value = 1)
  _X1.columns = ["P0:" + _ for _ in _X1.columns]
  m10 = logit(numpy.where(_Y == 0, 1, 0), _X1).fit(disp = 0)
  p10 = m10.params
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["MU:" + _ for _ in _X2.columns]
  m20 = negbinom2(_Y[_Y > 0], X2[_Y > 0]).fit(disp = 0)
  p20 = m20.params
  _X = _X1.join(_X2)
  return(zifnegbin2(_Y, _X))

