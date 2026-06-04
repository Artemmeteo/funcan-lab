# -*- coding: utf-8 -*-
"""Системы линейных функционалов и матрица признаков z_{i,k} = <x_i, phi_k>."""
import numpy as np
from quadrature import trapezoid_rule, inner


def trig_system(t, m):
    """Тригонометрическая ОНС в L2[0,1]."""
    funcs = [np.ones_like(t)]
    k = 1
    while len(funcs) < m:
        funcs.append(np.sqrt(2) * np.sin(2 * np.pi * k * t))
        if len(funcs) < m:
            funcs.append(np.sqrt(2) * np.cos(2 * np.pi * k * t))
        k += 1
    return np.array(funcs[:m])


def indicator_system(t, m):
    """Нормированные индикаторы m равных подотрезков: L_phi(x) = среднее x на подотрезке."""
    edges = np.linspace(t[0], t[-1], m + 1)
    funcs = []
    for k in range(m):
        ind = ((t >= edges[k]) & (t <= edges[k + 1])).astype(float)
        funcs.append(ind / trapezoid_rule(ind, t))
    return np.array(funcs)


def poly_system(t, m):
    """Полиномы ортонормированные в L2[0,1]."""
    P = np.array([t ** k for k in range(m)])
    Q = np.zeros_like(P)
    for k in range(m):
        v = P[k].copy()
        for j in range(k):
            v -= inner(P[k], Q[j], t) * Q[j]
        Q[k] = v / np.sqrt(inner(v, v, t))
    return Q


def features(X, Phi, t):
    """Матрица признаков: z_{i,k} = <x_i, phi_k>."""
    return trapezoid_rule(X[:, None, :] * Phi[None, :, :], t)


def gram(Phi, t):
    """Матрица Грама системы Phi (m,N):  G[k,l] = <phi_k, phi_l>."""
    return trapezoid_rule(Phi[:, None, :] * Phi[None, :, :], t)
