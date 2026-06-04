# -*- coding: utf-8 -*-
"""Метрика (евклидово расстояние) и метрики качества регрессии."""
import numpy as np


def dist_matrix(A, B):
    """Евклидовы расстояния между строками A (q,d) и B (l,d) -> матрица (q,l)."""
    a2 = np.sum(A ** 2, axis=1)[:, None]
    b2 = np.sum(B ** 2, axis=1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0))


def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot
