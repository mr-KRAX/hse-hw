import numpy as np
# from typing import override

from bandito.solvers import Solver


class EXP3(Solver):
  """EXP3
  
  Exponential-weight algorithm for Exploration and Exploitation
  """

  def __init__(self, bandit, gamma):
    super().__init__(bandit, 'EXP3')
    self.gamma = gamma
    self.weights = [1] * self.bandit.n
    self.k = self.bandit.n
    self.y = gamma

  def _recalculate_weight(self, i: int, p: float) -> None:
    """Recalculates handles weight according to reward

    Args:
      i (int): handle idx
      p (float): estimated probability
    """
    r = self.bandit.generate_reward(i)
    self.weights[i] *= np.exp((self.y * (r / p))
                              / self.k)

  # @override
  @property
  def estimated_probas(self):
    s = sum(self.weights)
    return [
        ((1-self.y) * w_i / s + self.y / self.k)
        for w_i in self.weights
    ]

  # @override
  def run_one_step(self):
    probas = self.estimated_probas
    i = np.random.choice(range(self.k), p=probas)
    self._recalculate_weight(i, probas[i])
    return i
