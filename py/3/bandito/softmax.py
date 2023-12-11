import numpy as np
# from typing import override

from bandito.solvers import Solver


class Softmax(Solver):
  """Softmax
  
  Selects an arm proportional to the amount of reward it has received before
  """

  def __init__(self, bandit, tau):
    super().__init__(bandit, 'Softmax')
    self.k = self.bandit.n
    self.weights = np.zeros(self.k, dtype=np.float32)
    self.tau = tau

  @staticmethod
  def _softmax(weights: np.array, tau):
    exp_values = np.exp(weights / tau)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

  def _recalculate_weight(self, i: int) -> None:
    """Recalculates handles weight according to reward

    Args:
      i (int): handle idx
    """
    n = self.counts[i]
    w = self.weights[i]
    r = self.bandit.generate_reward(i)
    self.weights[i] = ((n * w + r) / (n + 1.0))

  # @override
  @property
  def estimated_probas(self):
    return self._softmax(self.weights, self.tau)

  # @override
  def run_one_step(self):
    probas = self.estimated_probas
    i = np.random.choice(range(self.k), p=probas)
    self._recalculate_weight(i)
    return i
