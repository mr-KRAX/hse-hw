import time
import numpy as np
from scipy.stats import beta

from tqdm import tqdm_notebook

from bandito.bernoulli_bandit import BernoulliBandit


class Solver():
    """Solver

    Класс для имплементации решения проблемы с бандитами 
    """

    def __init__(self, bandit, name='<solver_name>'):
        """
        bandit (Bandit): Инициализация бандита.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = []  # Список id ручек от 0 до bandit.n-1.
        self.regret = 0.  # Суммарная ошибка.
        self.regrets = [0.]  # История суммарной ошибки.
        self.name = name

    def update_regret(self, i):
        # i (int): Индекс выбранной ручки.
        self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)

    # Dummy-метод оценки вероятностей (переопределяется для каждого solver'a)
    @property
    def estimated_probas(self) -> list[float]:
        raise NotImplementedError

    # Dummy-метод перехода на следующий шаг (переопределяется для каждого solver'a)
    def run_one_step(self) -> int:
        """Return the machine index to take action on."""
        raise NotImplementedError

    # Запуск работы бандита на num_steps шагов
    def run(self, num_steps):
        assert self.bandit is not None

        for _ in tqdm_notebook(range(num_steps), desc=self.name):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)


class EpsilonGreedy(Solver):
    """EpsilonGreedy
    
    epsilon-жадная стратегия
    """

    def __init__(self, bandit, eps, init_proba=1.0):
        """
        eps (float): Вероятность исследования случайной ручки.
        init_proba (float): начальное значение =  1.0;
        """
        # Сделали бандита
        super(EpsilonGreedy, self).__init__(bandit, 'EpsilonGreedy')

        assert 0. <= eps <= 1.0
        # Задали epsilon
        self.eps = eps

        self.estimates = [init_proba] * \
            self.bandit.n  # Optimistic initialization

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Дернули случайную ручку
            i = np.random.randint(0, self.bandit.n)
        else:
            # Выбрали наилучшую (на данный момент) ручку
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x])

        r = self.bandit.generate_reward(i)

        # Оценка для i-того бандита обновляется
        self.estimates[i] += 1. / \
            (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class UCB1(Solver):
    """UCB1

    UCB1 стратегия
    """

    def __init__(self, bandit, init_proba=1.0):
        super(UCB1, self).__init__(bandit, 'UCB1')
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Выбрать лучшую ручку с учетом UCB.
        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))
        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / \
            (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class BayesianUCB(Solver):
    """BayesianUCB
    
    Байесовская UCB стратегия.
    Предположим априорное Бета-распределение.
    """

    def __init__(self, bandit, c=2, init_a=1, init_b=1):
        """
        c (float): Сколько стандартных отклонений рассматривать в качестве UCB.
        init_a (int): Исходное значение a в Beta(a, b).
        init_b (int): Исходное значение b в Beta(a, b).
        """
        super(BayesianUCB, self).__init__(bandit, 'BayesianUCB')
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        # Выбрать лучшую ручку с учетом UCB.
        i = max(
            range(self.bandit.n),
            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x]) +
            beta.std(self._as[x], self._bs[x]) * self.c
        )
        r = self.bandit.generate_reward(i)

        # Обновление апостериорного бета-распределения
        self._as[i] += r
        self._bs[i] += (1 - r)

        return i


class ThompsonSampling(Solver):
    """ThompsonSampling
    
    Сэмплирование Томпсона
    """

    def __init__(self, bandit, init_a=1, init_b=1):
        """
        init_a (int): Исходное значение a в Beta(a, b).
        init_b (int): Исходное значение b в Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bandit, 'ThompsonSampling')

        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x])
                   for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)

        return i
