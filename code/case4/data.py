# -*- coding: utf-8 -*-
"""Синтетические функциональные данные (модель из условия Кейса 4)."""
import numpy as np
from quadrature import trapezoid_rule


def w_star(t):
    """
    Вычисляет истинную весовую функцию w*(t).

    Используемая весовая функция:
        w*(t) = 2 sin(2πt) - t

    Parameters
    t : ndarray
        Сетка значений аргумента на интервале наблюдения.

    Returns
    ndarray
        Значения функции w*(t) в точках сетки t.
    """
    return 2 * np.sin(2 * np.pi * t) - t


def make_dataset(n_samples, t, noise_x=0.05, noise_y=0.10, rng=None):
    """
    Генерирует синтетический набор функциональных данных.

    x_i(t) = a_i sin(2πt) + b_i cos(2πt) + c_i t + ε_i(t),
    y_i = <x_i, w*> + η_i.

    Parameters
    n_samples : int
        Количество генерируемых функций.

    t : ndarray
        Равномерная сетка точек на интервале наблюдения.

    noise_x : float, default=0.05
        Стандартное отклонение шума, добавляемого к функциям.

    noise_y : float, default=0.10
        Стандартное отклонение шума в целевой переменной.

    rng : numpy.random.Generator, optional
        Генератор случайных чисел. Если не задан, создаётся новый.

    Returns
    X : ndarray of shape (n_samples, N)
        Матрица значений функциональных наблюдений на сетке.

    y : ndarray of shape (n_samples,)
        Вектор целевых значений.
    """
    rng = np.random.default_rng() if rng is None else rng
    A = rng.normal(0, 1, n_samples)
    B = rng.normal(0, 1, n_samples)
    C = rng.normal(0, 1, n_samples)
    X = (A[:, None] * np.sin(2 * np.pi * t)[None, :]
         + B[:, None] * np.cos(2 * np.pi * t)[None, :]
         + C[:, None] * t[None, :])
    X = X + noise_x * rng.normal(size=X.shape)
    y = trapezoid_rule(X * w_star(t)[None, :], t) + noise_y * rng.normal(size=n_samples)
    return X, y


def make_piecewise_dataset(n_samples, t, n_seg=3, coefs=(3.0, -2.0, 1.0),
                           noise_x=0.05, noise_y=0.10, rng=None):
    """
    Генерирует набор кусочно-постоянных функциональных данных.

    Интервал наблюдения разбивается на n_seg подинтервалов.
    Для каждого наблюдения на каждом сегменте генерируется случайный
    уровень сигнала, после чего формируется кусочно-постоянная функция.

    m_k = (∫ x(t) I_k(t) dt) / (∫ I_k(t) dt),
    y = Σ β_k m_k + ε.

    Parameters
    n_samples : int
        Количество генерируемых наблюдений.

    t : ndarray
        Сетка точек на интервале наблюдения.

    n_seg : int, default=3
        Количество сегментов разбиения интервала.

    coefs : sequence of float, default=(3.0, -2.0, 1.0)
        Коэффициенты линейной модели для сегментных средних.

    noise_x : float, default=0.05
        Стандартное отклонение шума в функциях.

    noise_y : float, default=0.10
        Стандартное отклонение шума в целевой переменной.

    rng : numpy.random.Generator, optional
        Генератор случайных чисел.

    Returns
    X : ndarray of shape (n_samples, N)
        Матрица функциональных наблюдений.

    y : ndarray of shape (n_samples,)
        Вектор целевых значений.

    w : ndarray of shape (N,)
        Истинная весовая функция, соответствующая линейной зависимости
        между функциями и ответом.
    """
    rng = np.random.default_rng() if rng is None else rng
    coefs = np.asarray(coefs, float)
    edges = np.linspace(t[0], t[-1], n_seg + 1)
    seg = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_seg - 1)
    levels = rng.normal(0, 1, (n_samples, n_seg))
    X = levels[:, seg] + noise_x * rng.normal(size=(n_samples, t.size))
    means = np.empty((n_samples, n_seg))
    for k in range(n_seg):
        ind = ((t >= edges[k]) & (t <= edges[k + 1])).astype(float)
        means[:, k] = trapezoid_rule(X * ind[None, :], t) / trapezoid_rule(ind, t)
    y = means @ coefs + noise_y * rng.normal(size=n_samples)
    w = (coefs / (edges[1] - edges[0]))[seg]
    return X, y, w
