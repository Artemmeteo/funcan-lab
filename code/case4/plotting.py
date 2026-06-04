# -*- coding: utf-8 -*-
"""Настройка matplotlib (шрифт CMU Serif) и сохранение фигур/таблиц в report/figures/case4."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "report", "figures", "case4"))
FONTS = os.path.normpath(os.path.join(HERE, "..", "..", "report", "fonts"))
os.makedirs(OUT, exist_ok=True)

for _f in ["cmunrm.otf", "cmunbx.otf", "cmunti.otf"]:
    _p = os.path.join(FONTS, _f)
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "figure.dpi": 140, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.30, "font.size": 11, "axes.titlesize": 12,
})


def save(fig, name):
    """Сохранитяет фигуру как vector PDF в OUT."""
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def save_table(df, name):
    """Сохранитяет DataFrame как фрагмент LaTeX-таблицы (booktabs) в OUT."""
    tex = df.to_latex(index=False, escape=False, float_format="%.3f",
                      column_format="l" + "r" * (df.shape[1] - 1))
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(tex)
