import time
import numpy as np


class BernoulliBandit():
    """BernoulliBandit

    Общий класс для Бернуллиевских бандитов
    """

    def __init__(self, n: int, probas: list[float] = None, seed: int = None):
        assert probas is None or len(probas) == n

        # Кол-во ручек
        self.n = n
        if seed is None:
            seed = int(time.time())
        if probas is None:
            np.random.seed(seed)
            # Истинные вероятности ручек (случайно заданные)
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            # Истинные вероятности ручек, если заданы в функции
            self.probas = probas

        # Вероятность оптимальной ручки
        self.best_idx = max(range(n), key=lambda i: self.probas[i])
        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        """
        Генерация "выигрыша" для i-той ручки бандита
        """
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0
