
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DummySelector(TransformerMixin):
  '''DummySelector
  Selects the first N features from ids
  '''

  def __init__(self, ids, n=None):
    super().__init__()
    if n is None:
      self.ids = ids
    else:
      self.ids = ids[-n:]
    print('LOG_DEBUG: dummy ids', self.ids)

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    return X.copy()[:, self.ids]


class TSTransformer(DummySelector):
  '''TopWeightSelectTransformer
  Selects N the most weighty features according to passed w
  
  TODO: count wights inside fit 
  '''

  def __init__(self, n, w):
    self.ids = w.argsort()[-n:]
    print('LOG_DEBUG: top ids', self.ids)
