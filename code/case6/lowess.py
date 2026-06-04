# -*- coding: utf-8 -*-
"""Робастная схема LOWESS: итеративное перевзвешивание объектов по остаткам."""
import numpy as np
from metrics import dist_matrix


def lowess_weights(X, y, h, kernel, n_iter=5):
    """
    Вычисляет робастные веса LOWESS.

    На каждой итерации строится LOO-оценка Надарая–Ватсона
        a_i = Σ_j y_j gamma_j K(rho(x_i, x_j) / h) / Σ_j gamma_j K(rho(x_i, x_j) / h),
        gamma_i = ( 1 - (eps_i / (6 med{eps}))^2 )_+^2, eps_i = |a_i - y_i|.

    Возвращает (gamma, a): объектные веса после стабилизации и LOO-оценки a_i.
    """
    D = dist_matrix(X, X); Wbase = kernel(D / h); np.fill_diagonal(Wbase, 0.0)
    gamma = np.ones(len(y)); a = y.copy()
    for _ in range(n_iter):
        Wg = Wbase * gamma[None, :]
        den = Wg.sum(1); num = Wg @ y
        a = np.where(den > 0, num / np.where(den > 0, den, 1.0), y.mean())
        eps = np.abs(a - y); s = np.median(eps) + 1e-12
        gamma = np.maximum(0.0, 1.0 - (eps / (6.0 * s)) ** 2) ** 2
    return gamma, a


def lowess_predict(Xq, Xtr, ytr, gamma, h, kernel):
    """
    Выполняет прогноз робастной ядерной регрессии LOWESS.

    Использует оценку Надарая–Ватсона с объектными весами gamma:
        a(x) = Σ_i y_i gamma_i K(rho(x, x_i) / h) / Σ_i gamma_i K(rho(x, x_i) / h).
    """
    W = kernel(dist_matrix(Xq, Xtr) / h) * gamma[None, :]
    den = W.sum(1); num = W @ ytr
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), ytr.mean())
