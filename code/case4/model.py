# -*- coding: utf-8 -*-
"""Линейная модель: МНК через нормальные уравнения, ridge, метрики, весовая функция."""
import numpy as np
from numpy.linalg import solve


def design_matrix(Z):
    """Добавляет столбец единиц (свободный член)."""
    return np.hstack([np.ones((Z.shape[0], 1)), Z])


def ols_normal_equations(Z, y):
    """МНК через нормальные уравнения: beta = (X^T X)^{-1} X^T y."""
    Xd = design_matrix(Z)
    return solve(Xd.T @ Xd, Xd.T @ y)


def ridge_fit(Z, y, lam):
    """Гребневая регрессия: beta = (X^T X + lam I)^{-1} X^T y"""
    Xd = design_matrix(Z)
    Reg = np.eye(Xd.shape[1]); Reg[0, 0] = 0.0
    return solve(Xd.T @ Xd + lam * Reg, Xd.T @ y)


def predict(Z, beta):
    """Вычисляет прогнозы линейной модели."""
    return design_matrix(Z) @ beta


def metrics(y, yhat):
    """Возвращает (MSE, RMSE, R^2)."""
    err = y - yhat
    mse = float(np.mean(err ** 2))
    ss_res = float(np.sum(err ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return mse, float(np.sqrt(mse)), 1.0 - ss_res / ss_tot


def weight_function(beta, Phi):
    """w(t) = sum_k beta_k phi_k(t)."""
    return beta[1:] @ Phi
