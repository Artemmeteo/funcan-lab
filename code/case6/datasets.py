# -*- coding: utf-8 -*-
"""Реальные датасеты и оценка качества ядерной регрессии на них."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split

from kernels import KERNELS
from metrics import dist_matrix, mae, rmse, r2
from nadaraya_watson import nw_fixed, loo_fixed_from_D


def standardize(tr, te):
    """Стандартизует признаки по статистикам обучающей выборки."""
    mu, sd = tr.mean(0), tr.std(0) + 1e-12
    return (tr - mu) / sd, (te - mu) / sd


def load_real_datasets(subsample_california=2000):
    """
    Загружает реальные датасеты для задачи регрессии.

    Возвращает Diabetes и подвыборку California Housing.
    """
    data = {}
    db = load_diabetes()
    data["Diabetes"] = (db.data, db.target)
    try:
        ch = fetch_california_housing()
        sub = np.random.default_rng(0).choice(len(ch.target), size=subsample_california, replace=False)
        data["California Housing"] = (ch.data[sub], ch.target[sub])
    except Exception as e:
        print("  California Housing недоступен (нет сети?):", e)
    return data


def evaluate_dataset(Xraw, yraw, h_grid_q=np.linspace(0.3, 3.0, 14), seed=0):
    """
    Оценивает качество ядерной регрессии на датасете.

    Для каждого ядра подбирает оптимальную ширину окна h по LOO-контролю,
    после чего вычисляет метрики MAE, RMSE и R² на тестовой выборке.
    """
    Xtr, Xte, ytr, yte = train_test_split(Xraw, yraw, test_size=0.3, random_state=seed)
    Xtr, Xte = standardize(Xtr, Xte); ytr = np.asarray(ytr, float); yte = np.asarray(yte, float)
    Dtr = dist_matrix(Xtr, Xtr); h_grid = np.median(Dtr[Dtr > 0]) * h_grid_q
    res = []
    for name, ker in KERNELS.items():
        loo = [rmse(ytr, loo_fixed_from_D(Dtr, ytr, h, ker)) for h in h_grid]
        hb = h_grid[int(np.argmin(loo))]; pred = nw_fixed(Xte, Xtr, ytr, hb, ker)
        res.append([name, round(hb, 3), round(mae(yte, pred), 3), round(rmse(yte, pred), 3), round(r2(yte, pred), 3)])
    return pd.DataFrame(res, columns=["ядро", "$h^*$", "MAE", "RMSE", "$R^2$"])
