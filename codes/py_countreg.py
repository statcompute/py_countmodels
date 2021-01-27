# py_countreg/py_countreg.py
# exec(open('py_countreg/py_countreg.py').read())
# 0.0.2

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
  pr = numpy.exp(-mu) * numpy.float_power(mu, y) / scipy.special.factorial(y)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def stdpoisson(Y, X):
  """
  The function estimates a standard poisson regression.
  Parameters:
    Y : a pandas series for the frequency outcome with integer values.
    X : a pandas dataframe with model variables that are all numeric values.
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
       numpy.float_power(a1 / (a1 + mu), a1) * numpy.float_power(mu / (a1 + mu), y)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def negbinom2(Y, X):
  """
  The function estimates a negative binomial (NB-2) regression.
  Parameters:
    Y : a pandas series for the frequency outcome with integer values.
    X : a pandas dataframe with model variables that are all numeric values.
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
        start_params = numpy.append(p0, a0)
      return(super(negbinom2, self).fit(start_params = start_params, method = method,
                                        maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  p0 = stdpoisson(Y, X).fit(disp = 0).params
  a0 = 1
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
       numpy.float_power(_a + xi * y, y - 1)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def genpoisson(Y, X):
  """
  The function estimates a generalized poisson regression. In addition to regression
  coefficients, there is a scale parameter S such that Xi = Exp(S). In a generalized
  poisson distribution, the VAR(Y) = E(Y) / [(1 - Xi) ^ 2] such that [(1 - Xi) ^ 2] > 1
  indicates the under-dispersion and [(1 - Xi) ^ 2] < 1 indicates the over-dispersion. 
  Parameters:
    Y : a pandas series for the frequency outcome with integer values.
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

  p0 = stdpoisson(Y, X).fit(disp = 0).params
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
       (1 - p0) * numpy.exp(-mu) * numpy.float_power(mu, y) / \
       ((1 - numpy.exp(-mu)) * scipy.special.factorial(y)) * (1 - i0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def hdlpoisson(Y, X1, X2):
  """
  The function estimates a hurdle poisson regression, which is the composite 
  between point mess at zero and a zero-trucated poisson distribution.
  In the model outcome, estimated coefficients starting with "P0:" are used 
  to predict the probability of zero outcomes and estimated coefficients 
  starting with "MU:" are used to predict frequency outcomes for a zero-trucated
  poisson.
  Parameters:
    Y  : a pandas series for the frequency outcome with integer values, including zeros.
    X1 : a pandas dataframe with the probability model variables that are all numeric values.
    X2 : a pandas dataframe with the count model variables that are all numeric values.
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
        #start_params = numpy.concatenate([p10, numpy.zeros(_X2.shape[1])])
        start_params = numpy.concatenate([p10, p20])
      return(super(hdlpoisson, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X1 = X1.copy()
  _X2 = X2.copy()
  _X1.insert(loc = 0, column = "_CONST", value = 1)
  _X1.columns = ["P0:" + _ for _ in _X1.columns]
  p10 = logit(numpy.where(_Y == 0, 1, 0), _X1).fit(disp = 0).params
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["MU:" + _ for _ in _X2.columns]
  p20 = ztrpoisson(Y[Y > 0], X[Y > 0]).fit(disp = 0).params
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
       (1 - p0) * numpy.exp(-mu) * numpy.float_power(mu, y) / scipy.special.factorial(y) * (1 - i0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def zifpoisson(Y, X1, X2):
  """
  The function estimates a zero-inflated poisson regression, which is the 
  composite between point mess at zero and a standard poisson distribution.
  In the model outcome, estimated coefficients starting with "P0:" are used 
  to predict the probability of zero outcomes and estimated coefficients 
  starting with "MU:" are used to predict frequency outcomes for a standard
  poisson.
  Parameters:
    Y  : a pandas series for the frequency outcome with integer values, including zeros.
    X1 : a pandas dataframe with the probability model variables that are all numeric values.
    X2 : a pandas dataframe with the count model variables that are all numeric values.
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
  p10 = logit(numpy.where(_Y == 0, 1, 0), _X1).fit(disp = 0).params
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["MU:" + _ for _ in _X2.columns]
  p20 = ztrpoisson(Y[Y > 0], X[Y > 0]).fit(disp = 0).params
  _X = _X1.join(_X2)
  return(zifpoisson(_Y, _X))


#################### 06. Conway-Maxwell Poisson Regression ####################


def _ll_compoisson(y, x, beta, s):
  """
  The function calculates the log likelihood function of the Conway-Maxwell
  poisson regression.
  Parameters:
    y    : the frequency outcome.
    x    : variables in the conway-maxwell poisson regression
    beta : coefficients in the conway maxwell poisson regression
    s    : the scale parameter in the Conway-Maxwell distribution and is equal to log(nv)
  """

  mu = numpy.exp(numpy.dot(x, beta))
  nv = numpy.exp(s) 
  _z = 0
  for _n in range(100):
    _z = _z + numpy.float_power(mu, _n) / numpy.float_power(scipy.special.factorial(_n), nv)

  pr = numpy.float_power(mu, y) / numpy.float_power(scipy.special.factorial(y), nv) * numpy.float_power(_z, -1)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def compoisson(Y, X):
  """
  The function estimates a Conway-Maxwell poisson regression. 
  Given MU = exp(x * beta), E(Y) ~= MU + nv / 2 - 0.5. In addition to estimated 
  coefficients beta, there is a scaled parameter S such that nv = Exp(S). 
  In the COMpoisson, since VAR(Y) ~= E(Y) / nv, nv > 1 suggests the under-dispersion
  and nv < 1 suggests the over-dispersion. 
  Parameters:
    Y : a pandas series for the frequency outcome with integer values.
    X : a pandas dataframe with the probability model variables that are all numeric values.
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

  p0 = stdpoisson(Y, X).fit(disp = 0).params
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
       (1 - p0) / (1 - numpy.float_power(a1 / (a1 + mu), a1)) * \
       scipy.special.gamma(y + a1) / (scipy.special.gamma(y + 1) * scipy.special.gamma(a1)) * \
       numpy.float_power(a1 / (a1 + mu), a1) * numpy.float_power(mu / (a1 + mu), y) * (1 - i0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def hdlnegbin2(Y, X1, X2):
  """
  The function estimates a hurdle negative binomial regression, which is the 
  composite between point mess at zero and a zero-truncated negative binomial 
  distribution.
  In the model outcome, estimated coefficients starting with "P0:" are used 
  to predict the probability of zero outcomes and estimated coefficients 
  starting with "MU:" are used to predict frequency outcomes for a zero-trucated
  negative binomial.
  Parameters:
    Y  : a pandas series for the frequency outcome with integer values, including zeros.
    X1 : a pandas dataframe with the probability model variables that are all numeric values.
    X2 : a pandas dataframe with the count model variables that are all numeric values.
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
  p10 = logit(numpy.where(_Y == 0, 1, 0), _X1).fit(disp = 0).params
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["MU:" + _ for _ in _X2.columns]
  p20 = ztrnegbin2(Y[Y > 0], X[Y > 0]).fit(disp = 0).params
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
  pr = (p0 + (1 - p0) * numpy.float_power(a1 / (a1 + mu), a1)) * i0 + \
       (1 - p0) * scipy.special.gamma(y + a1) / (scipy.special.gamma(y + 1) * scipy.special.gamma(a1)) * \
       numpy.float_power(a1 / (a1 + mu), a1) * numpy.float_power(mu / (a1 + mu), y) * (1 - i0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def zifnegbin2(Y, X1, X2):
  """
  The function estimates a zero-inflated negative binomial regression, which is
  the composite between point mess at zero and a negative binomial distribution.
  In the model outcome, estimated coefficients starting with "P0:" are used to
  predict the probability of zero outcomes and estimated coefficients starting 
  with "MU:" are used to predict frequency outcomes for a standard negative
  binomial.
  Parameters:
    Y  : a pandas series for the frequency outcome with integer values, including zeros.
    X1 : a pandas dataframe with the probability model variables that are all numeric values.
    X2 : a pandas dataframe with the count model variables that are all numeric values.
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
  p10 = logit(numpy.where(_Y == 0, 1, 0), _X1).fit(disp = 0).params
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["MU:" + _ for _ in _X2.columns]
  p20 = ztrnegbin2(Y[Y > 0], X[Y > 0]).fit(disp = 0).params
  _X = _X1.join(_X2)
  return(zifnegbin2(_Y, _X))


#################### 09. Zero-truncated Poisson Regression ####################


def _ll_ztrpoisson(y, x, beta):
  """
  The function calculates the log likelihood function of the zero-truncated
  Poisson regression.
  Parameters:
    y    : the frequency outcome without zero
    x    : variables of the negative binomial regression
    beta : coefficients of the negative binomial regression
  """

  mu = numpy.exp(numpy.dot(x, beta))
  p0 = numpy.exp(-mu)
  pr = numpy.exp(-mu) * numpy.float_power(mu, y) / scipy.special.factorial(y) / (1 - p0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def ztrpoisson(Y, X):
  """
  The function estimates a zero-truncated Poisson regression.
  Parameters:
    Y : a pandas series for the frequency outcome wit non-zero integer values.
    X : a pandas dataframe with model variables that are all numeric values.
  """

  class ztrpoisson(gll):
    def __init__(self, endog, exog, **kwds):
      super(ztrpoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      beta = params
      ll = _ll_ztrpoisson(self.endog, self.exog, beta)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      if start_params == None:
        start_params = numpy.zeros(self.exog.shape[1])
      return(super(ztrpoisson, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  return(ztrpoisson(_Y, _X))


#################### 10. Zero-truncated Negative Binomial Regression ####################


def _ll_ztrnegbin2(y, x, beta, alpha):
  """
  The function calculates the log likelihood function of the zero-truncated 
  negative binomial (NB-2) regression.
  Parameters:
    y     : the frequency outcome with non-zero integer values. 
    x     : variables of the negative binomial regression
    beta  : coefficients of the negative binomial regression
    alpha : the dispersion parameter of the zero-truncated negative binomial regression
  """

  mu = numpy.exp(numpy.dot(x, beta))
  a1 = 1 / alpha
  p0 = numpy.float_power(a1 / (a1 + mu), a1)
  pr = scipy.special.gamma(y + a1) / (scipy.special.gamma(y + 1) * scipy.special.gamma(a1)) * \
       numpy.float_power(a1 / (a1 + mu), a1) * numpy.float_power(mu / (a1 + mu), y) / (1 - p0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def ztrnegbin2(Y, X):
  """
  The function estimates a zero-truncated negative binomial (NB-2) regression.
  Parameters:
    Y : a pandas series for the frequency outcome with non-zero integer values.
    X : a pandas dataframe with model variables that are all numeric values.
  """

  class ztrnegbin2(gll):
    def __init__(self, endog, exog, **kwds):
      super(ztrnegbin2, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      alpha = params[-1]
      beta = params[:-1]
      ll = _ll_ztrnegbin2(self.endog, self.exog, beta, alpha)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_ALPHA')
      if start_params == None:
        start_params = numpy.append(p0, 1)
      return(super(ztrnegbin2, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  p0 = ztrpoisson(Y, X).fit(disp = 0).params
  a0 = 1
  return(ztrnegbin2(_Y, _X))


#################### 11. Zero-truncated Generalized Poisson Regression ####################


def _ll_ztgpoisson(y, x, beta, s):
  """
  The function calculates the log likelihood function of the zero-truncated 
  generalized poisson regression.
  Parameters:
    y    : the frequency outcome with non-zero integer values. 
    x    : variables of the negative binomial regression
    beta : coefficients of the negative binomial regression
    s    : the scaled parameter of the zero-truncated generalized poisson regression
  """

  mu = numpy.exp(numpy.dot(x, beta))
  xi = numpy.exp(s)
  _a = mu * (1 - xi)
  p0 = numpy.exp(-_a)
  pr = _a / scipy.special.factorial(y) * numpy.exp(-_a - xi * y) * \
       numpy.float_power(_a + xi * y, y - 1) / (1 - p0)
  ll = numpy.log(pr)
  return(ll)

################################################################################

def ztgpoisson(Y, X):
  """
  The function estimates a zero-truncated Generalized Poisson regression. The
  scaled parameter S = Log(Xi). In the Generalized Poisson distribution, 
    VAR(Y) = E(Y) / [(1 - Xi) ^ 2]
  such that [(1 - Xi) ^ 2] > 1 means the under-dispersion and [(1 - Xi) ^ 2] < 1
  means the over-dispersion.
  Parameters:
    Y : a pandas series for the frequency outcome wit non-zero integer values.
    X : a pandas dataframe with model variables that are all numeric values.
  """

  class ztgpoisson(gll):
    def __init__(self, endog, exog, **kwds):
      super(ztgpoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      _s = params[-1]
      beta = params[:-1]
      ll = _ll_ztgpoisson(self.endog, self.exog, beta, _s)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_S')
      if start_params == None:
        start_params = numpy.append(p0, s0)
      return(super(ztgpoisson, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  p0 = ztrpoisson(Y, X).fit(disp = 0).params
  s0 = numpy.log(max(1e-4, 1 - numpy.power(numpy.mean(Y) / numpy.var(Y), 0.5)))
  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  return(ztgpoisson(_Y, _X))


#################### 12. Zero-truncated Conway-Maxwell Poisson Regression ####################


def _ll_ztcpoisson(y, x, beta, s):
  """
  The function calculates the log likelihood function of the zero-truncated
  conway-maxwell poisson regression.
  Parameters:
    y    : the frequency outcome with non-zero integer values.
    x    : variables of the negative binomial regression
    beta : coefficients of the negative binomial regression
    s    : the scaled parameter of the zero-truncated conway-maxwell poisson regression
  """

  mu = numpy.exp(numpy.dot(x, beta))
  nv = numpy.exp(s)
  _z = 0
  for _n in range(100):
    _z = _z + numpy.float_power(mu, _n) / numpy.float_power(scipy.special.factorial(_n), nv)

  pr = numpy.float_power(mu, y) / numpy.float_power(scipy.special.factorial(y), nv) * numpy.float_power(_z, -1) / \
       (1 - numpy.float_power(_z, -1))
  ll = numpy.log(pr)
  return(ll)

################################################################################

def ztcpoisson(Y, X):
  """
  The function estimates a zero-truncated Conway-Maxwell Poisson regression. 
  The scaled parameter S = Log(nv). In the Conway-Maxwell Poisson distribution,
    VAR(Y) ~= E(Y) / nv
  such that nv > 1 means the under-dispersion and nv < 1 means the over-dispersion.
  Parameters:
    Y : a pandas series for the frequency outcome wit non-zero integer values.
    X : a pandas dataframe with model variables that are all numeric values.
  """

  class ztcpoisson(gll):
    def __init__(self, endog, exog, **kwds):
      super(ztcpoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      _s = params[-1]
      beta = params[:-1]
      ll = _ll_ztcpoisson(self.endog, self.exog, beta, _s)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_S')
      if start_params == None:
        start_params = numpy.append(p0, s0)
      return(super(ztcpoisson, self).fit(start_params = start_params, method = method,
                                         maxiter = maxiter, maxfun = maxfun, **kwds))

  p0 = ztrpoisson(Y, X).fit(disp = 0).params
  s0 = numpy.log(numpy.mean(Y) / numpy.var(Y))
  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  return(ztcpoisson(_Y, _X))

