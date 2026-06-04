# -*- coding: utf-8 -*-
"""
Кейс 6 - главный скрипт. Запускает все эксперименты и сохраняет фигуры и таблицы
в report/figures/case6/.

Запуск:  py run.py     (из папки code/case6)
Модули:  kernels.py, metrics.py, nadaraya_watson.py, lowess.py, datasets.py, plotting.py
"""
import numpy as np

from kernels import KERNELS, k_gauss
from metrics import dist_matrix, rmse
from nadaraya_watson import loo_fixed_from_D, loo_variable, nw_fixed
from lowess import lowess_weights, lowess_predict
from datasets import load_real_datasets, evaluate_dataset
from plotting import plt, save, save_table

RNG = np.random.default_rng(7)

# --------------------------------------------------------------- синтетика 1D
n = 220
x = np.sort(RNG.uniform(0.0, 4 * np.pi, n)); f_true = np.sin(x); y = f_true + 0.30 * RNG.normal(size=n)
X = x[:, None]; grid = np.linspace(0.0, 4 * np.pi, 500); Xg = grid[:, None]

print("== Кейс 6 ==")

# Рис.1 - данные
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.scatter(x, y, s=14, alpha=0.6, label="наблюдения $y_i$")
ax.plot(grid, np.sin(grid), "k-", lw=2.0, label=r"истинная $f(x)=\sin x$")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title("Синтетическая задача: функция и наблюдения"); ax.legend()
save(fig, "fig1.pdf")

# Рис.2 - влияние h
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.scatter(x, y, s=12, alpha=0.35, color="gray", label="наблюдения")
ax.plot(grid, np.sin(grid), "k-", lw=2.0, label="истинная $f$")
for h, st in zip([0.15, 0.7, 3.0], ["C0--", "C2-", "C3:"]):
    ax.plot(grid, nw_fixed(Xg, X, y, h, k_gauss), st, lw=1.8, label=f"НВ, $h={h}$")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title("Влияние ширины окна $h$ (ядро Гаусса)"); ax.legend(fontsize=9)
save(fig, "fig2.pdf")

# Рис.3 - LOO по h и k
D = dist_matrix(X, X)
hs = np.linspace(0.08, 3.0, 40); loo_h = np.array([rmse(y, loo_fixed_from_D(D, y, h, k_gauss)) for h in hs])
h_best = hs[loo_h.argmin()]
ks = np.arange(2, 60); loo_k = np.array([rmse(y, loo_variable(X, y, int(k), k_gauss)) for k in ks])
k_best = ks[loo_k.argmin()]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(hs, loo_h, "o-", ms=3); ax[0].axvline(h_best, color="C3", ls="--", label=f"$h^*={h_best:.2f}$")
ax[0].set_xlabel("ширина окна $h$"); ax[0].set_ylabel("LOO RMSE"); ax[0].set_title("Фиксированное окно"); ax[0].legend()
ax[1].plot(ks, loo_k, "s-", ms=3, color="C2"); ax[1].axvline(k_best, color="C3", ls="--", label=f"$k^*={k_best}$")
ax[1].set_xlabel("число соседей $k$"); ax[1].set_ylabel("LOO RMSE"); ax[1].set_title("Переменное окно"); ax[1].legend()
save(fig, "fig3.pdf")

# Таблица 1 + Рис.4 - ядра
import pandas as pd
rows, preds = [], {}
for name, ker in KERNELS.items():
    loo = np.array([rmse(y, loo_fixed_from_D(D, y, h, ker)) for h in hs]); hb = hs[loo.argmin()]
    rows.append([name, round(hb, 4), round(loo.min(), 4)]); preds[name] = nw_fixed(Xg, X, y, hb, ker)
save_table(pd.DataFrame(rows, columns=["ядро", "$h^*$", "LOO RMSE"]).sort_values("LOO RMSE"), "table1.tex")
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.plot(grid, np.sin(grid), "k-", lw=2.2, label="истинная $f$")
for name in KERNELS:
    ax.plot(grid, preds[name], lw=1.4, label=name)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title(r"Прогноз при разных ядрах (каждое со своим $h^*$)"); ax.legend(fontsize=8)
save(fig, "fig4.pdf")

# Рис.5 + Рис.6 - выбросы и LOWESS
y_out = y.copy(); out_idx = RNG.choice(n, size=int(0.12 * n), replace=False)
y_out[out_idx] += RNG.normal(0, 4.0, size=out_idx.size)
h_rob = h_best
pred_nw = nw_fixed(Xg, X, y_out, h_rob, k_gauss)
gamma, a_loo = lowess_weights(X, y_out, h_rob, k_gauss, n_iter=5)
pred_lowess = lowess_predict(Xg, X, y_out, gamma, h_rob, k_gauss)
rmse_nw_out = rmse(np.sin(grid), pred_nw); rmse_lw_out = rmse(np.sin(grid), pred_lowess)
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.scatter(x, y_out, s=14, alpha=0.4, color="gray", label="наблюдения с выбросами")
ax.scatter(x[out_idx], y_out[out_idx], s=40, facecolors="none", edgecolors="C3", label="выбросы")
ax.plot(grid, np.sin(grid), "k-", lw=2.0, label="истинная $f$")
ax.plot(grid, pred_nw, "C0--", lw=1.8, label="НВ (обычное)")
ax.plot(grid, pred_lowess, "C2-", lw=2.0, label="LOWESS")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title("Обычное сглаживание против LOWESS"); ax.legend(fontsize=8)
save(fig, "fig5.pdf")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); mask = np.ones(n, bool); mask[out_idx] = False
ax[0].scatter(x[mask], gamma[mask], s=16, color="C2", label="обычные объекты")
ax[0].scatter(x[out_idx], gamma[out_idx], s=40, color="C3", label="выбросы")
ax[0].set_xlabel("$x$"); ax[0].set_ylabel(r"вес $\gamma_i$"); ax[0].set_title(r"Веса $\gamma_i$ после стабилизации LOWESS"); ax[0].legend(fontsize=9)
ax[1].hist(np.abs(a_loo - y_out)[mask], bins=20, alpha=0.7, label="обычные объекты")
ax[1].hist(np.abs(a_loo - y_out)[out_idx], bins=20, alpha=0.7, color="C3", label="выбросы")
ax[1].set_xlabel(r"остаток $|a_i-y_i|$"); ax[1].set_ylabel("число объектов"); ax[1].set_title("Распределение остатков"); ax[1].legend(fontsize=9)
save(fig, "fig6.pdf")

# Рис.7 - загрязнение
fracs = np.linspace(0.0, 0.40, 11); rmse_nw, rmse_lw = [], []
for fr in fracs:
    yy = y.copy()
    if fr > 0:
        oi = RNG.choice(n, size=max(1, int(fr * n)), replace=False); yy[oi] += RNG.normal(0, 4.0, size=oi.size)
    pnw = nw_fixed(Xg, X, yy, h_best, k_gauss)
    g, _ = lowess_weights(X, yy, h_best, k_gauss, n_iter=5); plw = lowess_predict(Xg, X, yy, g, h_best, k_gauss)
    rmse_nw.append(rmse(np.sin(grid), pnw)); rmse_lw.append(rmse(np.sin(grid), plw))
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.plot(fracs, rmse_nw, "o-", label="НВ (обычное)"); ax.plot(fracs, rmse_lw, "s-", label="LOWESS")
ax.set_xlabel("доля выбросов"); ax.set_ylabel(r"RMSE к истинной $f$"); ax.set_title("Загрязнение данных: НВ против LOWESS"); ax.legend()
save(fig, "fig7.pdf")

# Рис.8 - неоднородная плотность
xa = np.concatenate([RNG.uniform(0, 1.2, 90), RNG.uniform(1.2, 5.0, 40), RNG.uniform(5.0, 4 * np.pi, 90)])
xa = np.sort(xa); ya = np.sin(xa) + 0.30 * RNG.normal(size=xa.size); Xa = xa[:, None]; Da = dist_matrix(Xa, Xa)
loo_fix = np.array([rmse(ya, loo_fixed_from_D(Da, ya, h, k_gauss)) for h in hs]).min()
loo_var = np.array([rmse(ya, loo_variable(Xa, ya, int(k), k_gauss)) for k in np.arange(2, 60)]).min()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(xa, ya, s=12, alpha=0.5); ax[0].plot(grid, np.sin(grid), "k-", lw=1.5)
ax[0].set_title("Данные с неоднородной плотностью"); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")
ax[1].bar(["фикс. окно", "перем. окно"], [loo_fix, loo_var], color=["C0", "C2"])
ax[1].set_ylabel("лучшая LOO RMSE"); ax[1].set_title("Фикс. против перем. окна")
for i, v in enumerate([loo_fix, loo_var]):
    ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
save(fig, "fig8.pdf")

# Рис.9 - ядро или ширина
spread_h = loo_h.max() - loo_h.min()
loo_by_kernel = np.array([rmse(y, loo_fixed_from_D(D, y, h_best, ker)) for ker in KERNELS.values()])
spread_kernel = loo_by_kernel.max() - loo_by_kernel.min()
fig, ax = plt.subplots(figsize=(6.6, 4.1))
ax.bar(["варьируем $h$\n(ядро фикс.)", "варьируем ядро\n($h$ фикс.)"], [spread_h, spread_kernel], color=["C0", "C2"])
for i, v in enumerate([spread_h, spread_kernel]):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
ax.set_ylabel("размах LOO RMSE"); ax.set_title("Что сильнее влияет: ширина окна или ядро")
save(fig, "fig9.pdf")

# --------------------------------------------------------------- реальные данные
real = load_real_datasets()
for num, (name, (Xr, yr)) in zip([2, 3], real.items()):
    save_table(evaluate_dataset(Xr, yr), f"table{num}.tex")

# Рис.10 - LOO против 5-fold кросс-валидации (подбор h)
def cv_rmse(Xc, yc, h, kernel, n_folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(yc))
    folds = np.array_split(idx, n_folds)
    pred = np.empty(len(yc))
    for i in range(n_folds):
        val = folds[i]; trn = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        pred[val] = nw_fixed(Xc[val], Xc[trn], yc[trn], h, kernel)
    return rmse(yc, pred)

cv = np.array([cv_rmse(X, y, h, k_gauss) for h in hs]); h_cv = hs[cv.argmin()]
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.plot(hs, loo_h, "o-", ms=3, label=f"LOO ($h^*={h_best:.2f}$)")
ax.plot(hs, cv, "s-", ms=3, label=f"5-fold CV ($h^*={h_cv:.2f}$)")
ax.set_xlabel("ширина окна $h$"); ax.set_ylabel("RMSE"); ax.set_title("Подбор $h$: LOO и кросс-валидация")
ax.legend()
save(fig, "fig10.pdf")

# Рис.11 - НВ как минимум локально взвешенного МНК (парабола Q(theta) в точке x0=pi)
x0 = np.array([[np.pi]])
w_i = k_gauss(dist_matrix(x0, X)[0] / h_best)
theta_star = (w_i * y).sum() / w_i.sum()
thetas = np.linspace(y.min(), y.max(), 200)
Q = np.array([(w_i * (th - y) ** 2).sum() for th in thetas])
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.plot(thetas, Q, "C0-", lw=1.8, label=r"$Q(\theta;x_0)$")
ax.axvline(theta_star, color="C3", ls="--", label=r"$\theta^\star=a(x_0)$ (формула НВ)")
ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$Q(\theta;x_0)$")
ax.set_title(r"НВ как минимум локально взвешенного МНК ($x_0=\pi$)"); ax.legend()
save(fig, "fig11.pdf")


print(f"  оптимум: h*={h_best:.3f}, k*={k_best}")
print(f"  выбросы: RMSE НВ={rmse_nw_out:.4f}, LOWESS={rmse_lw_out:.4f}")
print(f"  неоднородная плотность: фикс={loo_fix:.3f}, перем={loo_var:.3f}")
print(f"  размах LOO: по h={spread_h:.3f}, по ядрам={spread_kernel:.3f}")
print(f"  LOO h*={h_best:.3f} vs 5-fold CV h*={h_cv:.3f}")
print(f"  проверка НВ=argmin Q: a(x0)={theta_star:.4f}, argmin по сетке={thetas[Q.argmin()]:.4f}")
print("  фигуры и таблицы сохранены в report/figures/case6")
