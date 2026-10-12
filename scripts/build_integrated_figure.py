#!/usr/bin/env python3
"""Build the manuscript figure connecting money, votes, lists, and digital spending.

The script uses only machine-readable tables included in the repository. It is
therefore suitable for the processed-data reproduction path.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VOTE_TABLES = ROOT / "analysis_votes" / "tabelas"
DIGITAL_TABLES = ROOT / "analysis_digital" / "tabelas"
PAPER_FIGURES = ROOT / "paper" / "figures"
RESULT_FIGURES = ROOT / "results" / "figures"

YEARS = [2016, 2020, 2024]
COLORS = {2016: "#0072B2", 2020: "#E69F00", 2024: "#009E73"}


def read_csv(name: str, directory: Path) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    return pd.read_csv(path)


def build() -> tuple[Path, Path]:
    summary = read_csv("resumo_pelotas_dinheiro_votos.csv", VOTE_TABLES)
    attenuation = read_csv("atenuacao_coeficiente_dinheiro.csv", VOTE_TABLES)
    cv_vote = read_csv("validacao_cruzada_pelotas_resumo.csv", VOTE_TABLES)
    digital_auc = read_csv("pelotas_diagnostico_auc.csv", DIGITAL_TABLES)
    categories = read_csv("pelotas_categorias_despesa_resumo.csv", DIGITAL_TABLES)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

    # (a) Revenue-vote association.
    ax = axes[0, 0]
    ax.plot(summary["ano"], summary["rho_receita_votos"], marker="o", linewidth=2,
            label="All candidates")
    ax.plot(summary["ano"], summary["rho_receita_votos_dentro_lista"], marker="s", linewidth=2,
            linestyle="--", label="Within electoral list")
    for _, row in summary.iterrows():
        ax.annotate(f"{row['rho_receita_votos']:.2f}",
                    (row["ano"], row["rho_receita_votos"]),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    ax.set_ylim(0.48, 0.72)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("a  Revenue remains strongly associated with votes", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    # (b) Attenuation of money coefficient after current votes and list strength.
    ax = axes[0, 1]
    x = np.arange(len(YEARS))
    width = 0.34
    base = attenuation.set_index("ano").loc[YEARS]
    ax.bar(x - width / 2, base["beta_dinheiro_sem_votos_pp"], width,
           label="Money only")
    ax.bar(x + width / 2, base["beta_dinheiro_com_votos_lista_pp"], width,
           label="+ current votes and list strength")
    for idx, year in enumerate(YEARS):
        att = 100 * base.loc[year, "atenuacao_votos_lista"]
        y = max(base.loc[year, "beta_dinheiro_sem_votos_pp"],
                base.loc[year, "beta_dinheiro_com_votos_lista_pp"])
        ax.text(idx, y + 0.45, f"{att:.1f}% attenuated", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, YEARS)
    ax.set_ylabel("Election probability change\n(pp per revenue doubling)")
    ax.set_ylim(0, 13.4)
    ax.set_title("b  Votes absorb most of the money–election association", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # (c) Predictive contrasts in Pelotas.
    ax = axes[1, 0]
    models = [
        "Viabilidade prévia",
        "Viabilidade prévia + dinheiro",
        "Votos correntes + lista",
        "Votos correntes + lista + dinheiro",
    ]
    model_labels = [
        "Prior viability",
        "Prior viability + money",
        "Current votes + list",
        "Current votes + list + money",
    ]
    offsets = {2020: -0.12, 2024: 0.12}
    markers = {2020: "o", 2024: "s"}
    for year in [2020, 2024]:
        sub = cv_vote[(cv_vote["ano"] == year) & (cv_vote["modelo"].isin(models))].set_index("modelo")
        yvals = [sub.loc[m, "auc_media"] for m in models]
        yerr = [sub.loc[m, "auc_desvio"] for m in models]
        xx = np.arange(len(models)) + offsets[year]
        ax.errorbar(xx, yvals, yerr=yerr, marker=markers[year], linewidth=1.5,
                    capsize=2.5, label=str(year), color=COLORS[year])
    ax.set_xticks(np.arange(len(models)), model_labels, rotation=20, ha="right")
    ax.set_ylim(0.62, 1.025)
    ax.set_ylabel("Repeated cross-validated AUC")
    ax.set_title("c  Money improves prediction from prior viability\n    but not from post-election information", loc="left", fontweight="bold", fontsize=9.2)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    # (d) Digital expenditure compared with total and traditional advertising.
    ax = axes[1, 1]
    indicators = ["Despesa total", "Despesa não digital", "Despesa digital"]
    labels = ["Total expenditure", "Non-digital expenditure", "Strict digital expenditure", "Traditional advertising"]
    auc_matrix: dict[int, list[float]] = {}
    for year in YEARS:
        diag = digital_auc[digital_auc["ano"] == year].set_index("indicador")
        traditional = categories[(categories["ano"] == year) &
                                 (categories["grupo_despesa"] == "Publicidade tradicional")]
        if traditional.empty:
            raise ValueError(f"Traditional advertising row missing for {year}")
        auc_matrix[year] = [float(diag.loc[i, "auc"]) for i in indicators] + [float(traditional.iloc[0]["auc_eleicao"])]
    x = np.arange(len(labels))
    width = 0.24
    for j, year in enumerate(YEARS):
        ax.bar(x + (j - 1) * width, auc_matrix[year], width, label=str(year), color=COLORS[year])
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUC for election")
    ax.set_title("d  Digital spending contributes, but does not dominate", loc="left", fontweight="bold", fontsize=9.2)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = PAPER_FIGURES / "figure_6_money_votes_digital_mechanisms.pdf"
    png = PAPER_FIGURES / "figure_6_money_votes_digital_mechanisms.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(pdf, RESULT_FIGURES / pdf.name)
    shutil.copy2(png, RESULT_FIGURES / png.name)
    return pdf, png


if __name__ == "__main__":
    outputs = build()
    print("Built:")
    for output in outputs:
        print(output)
