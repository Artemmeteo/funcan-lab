# -*- coding: utf-8 -*-
"""
Кейс 4 - главный скрипт. Запускает все эксперименты и сохраняет фигуры и таблицы
в report/figures/case4/.

Запуск:  py run.py     (из папки code/case4)
Модули:  quadrature.py, data.py, functionals.py, model.py, plotting.py
"""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

from quadrature import inner, trapezoid_rule
from data import make_dataset, make_piecewise_dataset, w_star
from functionals import (trig_system, indicator_system, poly_system, features, gram)
from model import (design_matrix, ols_normal_equations, ridge_fit, predict,
                   metrics, weight_function)
from plotting import plt, save, save_table

RNG = np.random.default_rng(42)

# --------------------------------------------------------------- данные
a, b, N = 0.0, 1.0, 200
t = np.linspace(a, b, N)
w_true = w_star(t)

X_all, y_all = make_dataset(400, t, rng=RNG)
idx = RNG.permutation(X_all.shape[0]); tr, te = idx[:300], idx[300:]
X_train, y_train = X_all[tr], y_all[tr]
X_test,  y_test  = X_all[te], y_all[te]

print("== Кейс 4 ==")

# Рис.1 - примеры функций
fig, ax = plt.subplots(figsize=(7.0, 4.1))
for i in range(6):
    ax.plot(t, X_train[i], lw=1.2, alpha=0.85, label=f"$x_{{{i+1}}}(t)$")
ax.set_xlabel("$t$"); ax.set_ylabel("$x_i(t)$")
ax.set_title("Примеры функциональных наблюдений $x_i(t)$"); ax.legend(ncol=3, fontsize=8)
save(fig, "fig1.pdf")

# Рис.2 - сходимость трапеций
# берём непериодическую гладкую функцию e^t: для периодических функций трапеции
# на периоде дают спектральную точность и наклон O(N^-2) не виден
exact = float(np.e - 1.0); Ns = np.array([4, 8, 16, 32, 64, 128, 256, 512])
errs = np.array([abs(trapezoid_rule(np.exp(np.linspace(0, 1, n)), np.linspace(0, 1, n)) - exact)
                 for n in Ns])
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.loglog(Ns, errs, "o-", label="ошибка трапеций")
ax.loglog(Ns, errs[0] * (Ns / Ns[0]) ** -2.0, "--", label=r"наклон $\propto N^{-2}$")
ax.set_xlabel("число узлов $N$"); ax.set_ylabel("|ошибка интеграла|")
ax.set_title("Сходимость формулы трапеций"); ax.legend()
save(fig, "fig2.pdf")

# Рис.3 - матрица Грама
Phi_trig = trig_system(t, 7); G = gram(Phi_trig, t)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(G, cmap="viridis", vmin=-0.1, vmax=1.0)
ax.set_title("Матрица Грама тригонометрической системы")
ax.set_xlabel(r"$\ell$"); ax.set_ylabel("$k$")
fig.colorbar(im, fraction=0.046, pad=0.04)
save(fig, "fig3.pdf")
gram_dev = float(np.abs(G - np.eye(7)).max())

# линейность
phi = trig_system(t, 5)[2]; x1, x2 = X_train[0], X_train[1]; al, be = 1.7, -0.4
lin_resid = abs(inner(al*x1 + be*x2, phi, t) - (al*inner(x1, phi, t) + be*inner(x2, phi, t)))

# Таблица 1 - матрица Z
Phi_trig9 = trig_system(t, 9)
Z_train = features(X_train, Phi_trig9, t); Z_test = features(X_test, Phi_trig9, t)
cols = ["$1$", r"$\sqrt2\sin$", r"$\sqrt2\cos$", r"$\sqrt2\sin2$", r"$\sqrt2\cos2$",
        r"$\sqrt2\sin3$", r"$\sqrt2\cos3$", r"$\sqrt2\sin4$", r"$\sqrt2\cos4$"]
save_table(pd.DataFrame(np.round(Z_train[:5], 3), columns=cols), "table1.tex")

# МНК
beta_ne = ols_normal_equations(Z_train, y_train)
beta_ls, *_ = lstsq(design_matrix(Z_train), y_train, rcond=None)
ols_match = bool(np.allclose(beta_ne, beta_ls))
m_tr = metrics(y_train, predict(Z_train, beta_ne)); m_te = metrics(y_test, predict(Z_test, beta_ne))

# Рис.4 - восстановление w(t)
w_trig = weight_function(beta_ne, Phi_trig9)
Phi_ind9 = indicator_system(t, 9)
beta_ind = ols_normal_equations(features(X_train, Phi_ind9, t), y_train)
w_ind = weight_function(beta_ind, Phi_ind9)
beta_poly9 = ols_normal_equations(features(X_train, poly_system(t, 9), t), y_train)
w_poly = weight_function(beta_poly9, poly_system(t, 9))
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.plot(t, w_true, "k-", lw=2.5, label=r"истинная $w^\star(t)=2\sin 2\pi t - t$")
ax.plot(t, w_trig, "C0--", lw=1.8, label="тригонометрия, $m=9$")
ax.plot(t, w_ind, "C3-.", lw=1.6, label="средние по подотрезкам, $m=9$")
ax.plot(t, w_poly, "C2:", lw=1.4, label="полиномы, $m=9$")
ax.set_xlabel("$t$"); ax.set_ylabel("$w(t)$")
ax.set_title("Восстановление весовой функции (представителя Рисса)"); ax.legend(fontsize=9)
save(fig, "fig4.pdf")

# Таблица 2 - сравнение систем
systems = {"тригонометрия": trig_system, "средние по подотрезкам": indicator_system,
           "полиномы (ортонорм.)": poly_system}
m_cmp = 9; rows = []
for name, builder in systems.items():
    Phi = builder(t, m_cmp); Zte = features(X_test, Phi, t)
    beta = ols_normal_equations(features(X_train, Phi, t), y_train)
    mse_te, rmse_te, r2_te = metrics(y_test, predict(Zte, beta))   # метрики на тесте
    rows.append([name, round(mse_te, 4), round(rmse_te, 4), round(r2_te, 4)])
save_table(pd.DataFrame(rows, columns=["система ($m=9$)", "MSE", "RMSE", "$R^2$"]), "table2.tex")

# Рис.5 - устойчивость коэффициентов (бутстрэп)
def bootstrap_coef_std(builder, m, n_boot=200):
    Ztr = features(X_train, builder(t, m), t); n = Ztr.shape[0]; coefs = []
    for _ in range(n_boot):
        bi = RNG.integers(0, n, n)
        coefs.append(lstsq(design_matrix(Ztr[bi]), y_train[bi], rcond=None)[0])
    return np.std(np.array(coefs), axis=0)[1:]

fig, ax = plt.subplots(figsize=(7.0, 4.1)); width = 0.27; xpos = np.arange(m_cmp)
for j, (name, builder) in enumerate(systems.items()):
    ax.bar(xpos + (j - 1) * width, bootstrap_coef_std(builder, m_cmp), width=width, label=name, alpha=0.85)
ax.set_xlabel("номер коэффициента $k$"); ax.set_ylabel(r"бутстрэп-$\mathrm{std}(\hat\beta_k)$")
ax.set_title("Устойчивость коэффициентов разных систем"); ax.legend(fontsize=9)
save(fig, "fig5.pdf")

# Рис.6 - влияние m (малая обучающая выборка n=40, чтобы был виден эффект переобучения)
Xs, ys = X_train[:40], y_train[:40]
m_grid = list(range(1, 31))
fig, ax = plt.subplots(figsize=(7.0, 4.1))
for name, builder in [("тригонометрия", trig_system), ("средние по подотрезкам", indicator_system)]:
    tr_err, te_err = [], []
    for m in m_grid:
        Ztr = features(Xs, builder(t, m), t); Zte = features(X_test, builder(t, m), t)
        beta = lstsq(design_matrix(Ztr), ys, rcond=None)[0]
        tr_err.append(metrics(ys, predict(Ztr, beta))[1])
        te_err.append(metrics(y_test, predict(Zte, beta))[1])
    ax.plot(m_grid, tr_err, "--", label=f"{name}: train")
    ax.plot(m_grid, te_err, "-", label=f"{name}: test")
ax.set_xlabel("число функционалов $m$"); ax.set_ylabel("RMSE")
ax.set_title("Качество в зависимости от числа функционалов $m$"); ax.legend(fontsize=8)
save(fig, "fig6.pdf")

# Рис.7 - ridge
Phi = trig_system(t, 15); Ztr = features(X_train, Phi, t); Zte = features(X_test, Phi, t)
lams = np.logspace(-6, 2, 40); rmse_te_ridge, beta_norm = [], []
for lam in lams:
    beta = ridge_fit(Ztr, y_train, lam)
    rmse_te_ridge.append(metrics(y_test, predict(Zte, beta))[1]); beta_norm.append(np.linalg.norm(beta[1:]))
rmse_ols = metrics(y_test, predict(Zte, ridge_fit(Ztr, y_train, 0.0)))[1]
best = int(np.argmin(rmse_te_ridge))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].semilogx(lams, rmse_te_ridge, "o-", label="ridge, тест")
ax[0].axhline(rmse_ols, color="C3", ls="--", label=r"МНК ($\lambda=0$)")
ax[0].axvline(lams[best], color="gray", ls=":")
ax[0].set_xlabel(r"$\lambda$"); ax[0].set_ylabel("RMSE (тест)"); ax[0].set_title(r"Качество ridge от $\lambda$"); ax[0].legend(fontsize=9)
ax[1].loglog(lams, beta_norm, "s-", color="C2")
ax[1].set_xlabel(r"$\lambda$"); ax[1].set_ylabel(r"$\|\hat\beta_{1:}\|_2$"); ax[1].set_title(r"Норма коэффициентов от $\lambda$")
save(fig, "fig7.pdf")

# Рис.8 - устойчивость к шуму и сетке
Phi_fix = trig_system(t, 9)
noise_levels = np.linspace(0.0, 0.6, 13); rmse_vs_noise = []
for s in noise_levels:
    Xn, yn = make_dataset(400, t, noise_y=s, rng=np.random.default_rng(7))
    Ztr = features(Xn[:300], Phi_fix, t); Zte = features(Xn[300:], Phi_fix, t)
    beta = lstsq(design_matrix(Ztr), yn[:300], rcond=None)[0]
    rmse_vs_noise.append(metrics(yn[300:], predict(Zte, beta))[1])
N_grid = [16, 32, 64, 128, 256, 512]; rmse_vs_N = []
for n in N_grid:                              # усредняем по 5 датасетам, чтобы убрать случайный разброс
    tn = np.linspace(0, 1, n); Phi_n = trig_system(tn, 9); errs_n = []
    for seed in range(5):
        Xn, yn = make_dataset(400, tn, noise_y=0.10, rng=np.random.default_rng(seed))
        Ztr = features(Xn[:300], Phi_n, tn); Zte = features(Xn[300:], Phi_n, tn)
        beta = lstsq(design_matrix(Ztr), yn[:300], rcond=None)[0]
        errs_n.append(metrics(yn[300:], predict(Zte, beta))[1])
    rmse_vs_N.append(float(np.mean(errs_n)))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(noise_levels, rmse_vs_noise, "o-")
ax[0].set_xlabel(r"уровень шума отклика $\sigma_\eta$"); ax[0].set_ylabel("RMSE (тест)"); ax[0].set_title("Устойчивость к шуму")
ax[1].plot(N_grid, rmse_vs_N, "s-", color="C2")
ax[1].set_xlabel("число узлов сетки $N$"); ax[1].set_ylabel("RMSE (тест)")
ax[1].set_title("Устойчивость к изменению сетки"); ax[1].set_xscale("log", base=2)
ax[1].set_ylim(0.0, 0.16); ax[1].axhline(0.10, color="gray", ls=":", lw=1, label="шумовой порог")
ax[1].legend(fontsize=9)
save(fig, "fig8.pdf")

# Рис.9 - кусочно-гладкие данные: где средние по подотрезкам бьют тригонометрию
Xp, yp, w_pw = make_piecewise_dataset(400, t, rng=np.random.default_rng(123))
Xp_tr, yp_tr, Xp_te, yp_te = Xp[:300], yp[:300], Xp[300:], yp[300:]
pw_rmse = {}
for name, builder, m in [("тригонометрия (m=15)", trig_system, 15),
                         ("средние, m=3", indicator_system, 3),
                         ("полиномы (m=9)", poly_system, 9)]:
    Ztr = features(Xp_tr, builder(t, m), t); Zte = features(Xp_te, builder(t, m), t)
    beta = lstsq(design_matrix(Ztr), yp_tr, rcond=None)[0]
    pw_rmse[name] = metrics(yp_te, predict(Zte, beta))[1]
w_pw_ind = weight_function(ols_normal_equations(features(Xp_tr, indicator_system(t, 3), t), yp_tr),
                           indicator_system(t, 3))
w_pw_tr = weight_function(lstsq(design_matrix(features(Xp_tr, trig_system(t, 15), t)), yp_tr, rcond=None)[0],
                          trig_system(t, 15))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i in range(4):
    ax[0].plot(t, Xp_tr[i], lw=1.0, alpha=0.75)
ax[0].set_title("Кусочно-гладкие наблюдения"); ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$x_i(t)$")
ax[1].plot(t, w_pw, "k-", lw=2.5, label=r"истинная $w^\star$ (ступенчатая)")
ax[1].plot(t, w_pw_ind, "C2--", lw=1.8, label="средние, $m=3$")
ax[1].plot(t, w_pw_tr, "C0-.", lw=1.4, label="тригонометрия, $m=15$")
ax[1].set_title("Восстановление $w$ на кусочных данных"); ax[1].set_xlabel("$t$"); ax[1].legend(fontsize=8)
save(fig, "fig9.pdf")

# --------------------------------------------------------------- ключевые числа
print(f"  совпадение МНК/lstsq: {ols_match}")
print(f"  макс. отклонение Грама от I: {gram_dev:.2e};  невязка линейности: {lin_resid:.2e}")
print(f"  RMSE train/test (m=9): {m_tr[1]:.4f} / {m_te[1]:.4f};  R2 test = {m_te[2]:.4f}")
print(f"  ridge: RMSE МНК={rmse_ols:.4f}, лучший ridge={rmse_te_ridge[best]:.4f} при lambda={lams[best]:.3g}")
print("  кусочно-гладкие данные RMSE(test):", {k: round(v, 4) for k, v in pw_rmse.items()})
print("  фигуры и таблицы сохранены в report/figures/case4")
