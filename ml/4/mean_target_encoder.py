

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MeanTargetEncoder(BaseEstimator, TransformerMixin):
  """MeanTargetEncoder.

  Replaces categorical features with the mean target value for
  each category.

  Inspired by Brendan Hasz.
  https://brendanhasz.github.io/2019/03/04/target-encoding.html
  """

  def __init__(self, columns=None, noise=False):
    self.noise = noise
    if isinstance(columns, str):
      self.cols = [columns]
    else:
      self.cols = columns

  def _collect_and_validate_columns(self, X):
    if self.cols is None:
      # Get cat columns if it is not specified
      self.cols = [col for col in X
                   if str(X[col].dtype) == 'object']
    else:
      # Validate specified columns
      for col in self.cols:
        if col not in X:
          raise ValueError('Column \''+col+'\' not in X')

  def fit(self, X, y):
    self._collect_and_validate_columns(X)

    # column -> value -> mean target
    self.transformers = dict()

    for col in self.cols:
      col_transformer = dict()
      values = X[col].unique()
      for val in values:
        col_transformer[val] = y[X[col] == val].mean()
      self.transformers[col] = col_transformer

    return self

  def transform(self, X):
    X_copy = X.copy()
    for col, transformer in self.transformers.items():
      encoded = np.full(X.shape[0], np.nan)
      for val, mean_target in transformer.items():
        encoded[X[col] == val] = mean_target
      if self.noise:
        encoded += np.random.normal(0, 0.01, encoded.shape[0])
      X_copy[col] = encoded
    return X_copy



class MeanTargetSmoothedEncoder(MeanTargetEncoder):
  """MeanTargetEncoder with smoothing.

  Replaces categorical features with the smoothed mean target value for
  each category.
  """

  def __init__(self, C, columns=None, noise=False):
    super().__init__(columns, noise)
    self.C = C

  def fit(self, X, y):
    self._collect_and_validate_columns(X)

    # column -> value -> smoothed mean target
    self.transformers = dict()
    for col in self.cols:
      glob_mean = y.mean() * self.C
      col_transformer = dict()
      values = X[col].unique()
      for val in values:
        y_val = y[X[col] == val]
        col_transformer[val] = ((y_val.sum() + glob_mean) /
                                (y_val.count() + self.C))
      self.transformers[col] = col_transformer

    return self


class MeanCumsumTargetEncoder(MeanTargetEncoder):
  """MeanTargetEncoder counting min from above values.

  Replaces categorical features with the cumsum-mean target value for
  each category.
  """

  def __init__(self, columns=None, noise=False):
    # raise NotImplementedError(
    #     'MeanCumsumTargetEncoder author encountered deadline...')
    super().__init__(columns, noise)

  def fit(self, X, y):
    self._collect_and_validate_columns(X)

    # column -> values
    self.transformers = dict()
    for col in self.cols:
      X_copy = X[[col]].copy()
      X_copy['target'] = y
      sorted = X_copy.sort_values(col)
      cumsum = sorted['target'].cumsum()
      sorted['target'] = cumsum / (sorted.index + 1)
      self.transformers[col] = sorted['target']
      
    return self

  def transform(self, X):
    X_copy = X.copy()
    for col, tr in self.transformers.items():
      encoded = tr.copy()
      if self.noise:
        encoded += np.random.normal(0, 0.01, encoded.shape[0])
      X_copy[col] = encoded
    return X_copy
  