# -*- coding: utf-8 -*-
"""Численное интегрирование и скалярное произведение в L2[a,b]."""
import numpy as np


def trapezoid_rule(F, t):
    """
    Вычисляет определённый интеграл методом составной формулы трапеций.

    Parameters
    F : ndarray
        Значения подынтегральной функции на сетке.

    t : ndarray
        Узлы сетки.

    Returns
    ndarray
        Приближённое значение интеграла.
    """
    dt = np.diff(t)
    return np.sum((F[..., 1:] + F[..., :-1]) * 0.5 * dt, axis=-1)


def inner(F, G, t):
    """
    Вычисляет скалярное произведение функций в пространстве L2.

    Parameters
    F : ndarray
        Первая функция.

    G : ndarray
        Вторая функция.

    t : ndarray
        Узлы сетки.

    Returns
    ndarray
        Значение скалярного произведения.
    """
    return trapezoid_rule(F * G, t)
