# -*- coding: utf-8 -*-
"""Ядра для ядерного сглаживания. Аргумент r = rho/h >= 0; (u)_+ = max(0,u)."""
import numpy as np


def k_gauss(r): return np.exp(-0.5 * r ** 2)            # гауссовское (неограниченный носитель)
def k_epan(r):  return np.maximum(0.0, 1.0 - r ** 2)    # Епанечникова
def k_quart(r): return np.maximum(0.0, 1.0 - r ** 2) ** 2  # квадратичное (биквадратное)
def k_tri(r):   return np.maximum(0.0, 1.0 - r)         # треугольное


KERNELS = {"гауссовское": k_gauss, "Епанечникова": k_epan,
           "квадратичное": k_quart, "треугольное": k_tri}
