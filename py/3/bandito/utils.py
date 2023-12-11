import matplotlib.pyplot as plt
import numpy as np

from bandito.solvers import Solver
from bandito.bernoulli_bandit import BernoulliBandit


def plot_results(solvers: list[Solver], solver_names: list[str]) -> None:
    """
    Отрисовка результатов различных стратегий для бандитов.
    Args:
        solvers (list<Solver>): Список решений проблемы бандитов, которые нужно подгонять.
        solver_names (list<str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)

    # Рисунок. 1: Зависимость ошибки от времени.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Время')
    ax1.set_ylabel('Накопленная ошибка')
    ax1.grid('k', ls='--', alpha=0.3)
    ax1.set_title('Накопленная ошибка от итерации')

    # Рисунок. 2: Вероятности, оцененные алгоритмами.
    sorted_indices = sorted(range(b.n), key=lambda x: b.probas[x])
    ax2.plot(range(b.n), [b.probas[x]
             for x in sorted_indices], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probas[x]
                 for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Ручки, отсортированные по ' + r'$\theta$')
    ax2.set_ylabel('Оцененная алгоритмом ' + r'$\hat\theta$')
    ax2.grid('k', ls='--', alpha=0.3)
    ax2.set_title('Оригинальные и оценочные вероятности ручек')

    ax4 = fig.add_subplot(223)
    n = float(len(solvers[0].regrets))
    handles = range(b.n)
    bottom = np.zeros(b.n)
    for i, s in enumerate(solvers):
        
        ax4.bar(handles, np.array(s.counts) / n, 0.85, label=solver_names[i], bottom=bottom)
        bottom += np.array(s.counts) / n

    ax4.set_xticks(handles)
    ax4.set_xlabel('Ручки')
    ax4.set_ylabel('Доля каждой ручки в общем кол-ве действий')
    ax4.grid('k', ls='--', alpha=0.3)
    ax4.set_title('Доли каждой ручки в общем кол-ве действий')
    ax4.plot(b.best_idx, 0, 'wo')
    
    
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25))
    
    
    # # Рисунок 3: Срабатывание ручек
    # for s in solvers:
    #     ax3.plot(range(b.n), np.array(s.counts) /
    #              float(len(solvers[0].regrets)), ds='steps', lw=2)
    # ax3.set_xlabel('Ручки')
    # ax3.set_ylabel('Доля каждой ручки в общем кол-ве действий')
    # ax3.grid('k', ls='--', alpha=0.3)
    # ax3.set_title('Доли каждой ручки в общем кол-ве действий')

    plt.show()


def experiment(N, bandit: BernoulliBandit, solvers: dict[str, Solver]) -> None:
    """Прогонка эксперимента с бернуллиевским бандитом с K руками,
    в каждой из которых случайно задается вероятность выигрыша

    Args:
        N (_type_):Кол-во испытаний.
        bandit (BernoulliBandit): Модель бандита
        solvers (dict[str, Solver]): Список (имя -> модель) моделей 
            решения задачи о Бандите(эпсилон-жадная, UCB и тд)
    """

    print("Истинные вероятности выигрыша у Бернуллиевского бандита:\n", bandit.probas)
    print("У лучшей ручки индекс: {} и вероятность: {}".format(
        bandit.best_idx, bandit.best_proba))

    for _, s in solvers.items():
        s.run(N)

    plot_results(list(solvers.values()), list(solvers.keys()))
