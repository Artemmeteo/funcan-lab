# -*- coding: utf-8 -*-
"""Оценка Надарая-Ватсона: фиксированное и переменное окно, LOO-прогнозы."""
import numpy as np
from metrics import dist_matrix


def _smooth(W, ytr):
    den = W.sum(1); num = W @ ytr
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), ytr.mean())


def nw_fixed(Xq, Xtr, ytr, h, kernel):
    """Прогноз НВ с фиксированным окном h."""
    return _smooth(kernel(dist_matrix(Xq, Xtr) / h), ytr)


def nw_variable(Xq, Xtr, ytr, k, kernel):
    """Прогноз НВ с переменным окном: h(x) = расстояние до (k+1)-го соседа."""
    D = dist_matrix(Xq, Xtr); Ds = np.sort(D, axis=1)
    hloc = Ds[:, min(k, Ds.shape[1] - 1)]; hloc = np.where(hloc > 0, hloc, 1.0)
    return _smooth(kernel(D / hloc[:, None]), ytr)


def loo_fixed_from_D(D, y, h, kernel):
    """LOO-прогноз НВ (фикс. окно) по готовой матрице расстояний D (l,l)."""
    W = kernel(D / h); np.fill_diagonal(W, 0.0)
    return _smooth(W, y)


def loo_variable(X, y, k, kernel):
    """LOO-прогноз НВ (перем. окно): окно = расстояние до k-го
    ближайшего соседа после исключения самого объекта."""
    D = dist_matrix(X, X).copy(); np.fill_diagonal(D, np.inf); Ds = np.sort(D, axis=1)
    hloc = Ds[:, min(k, Ds.shape[1] - 2)]; hloc = np.where(np.isfinite(hloc) & (hloc > 0), hloc, 1.0)
    return _smooth(kernel(D / hloc[:, None]), y)
