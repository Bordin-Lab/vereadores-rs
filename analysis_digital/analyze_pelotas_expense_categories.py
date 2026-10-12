"""Composição das despesas e eleição em Pelotas.

Produz uma decomposição candidato a candidato das rubricas harmonizadas,
permitindo identificar quais tipos de gasto reproduzem a ordenação observada
entre eleitos e não eleitos.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-categorias-pelotas")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TABLE_OUT = OUT / "tabelas"
FIGURE_OUT = OUT / "figuras"
PANEL_PATH = OUT / "dados" / "painel_candidatos_despesas_digitais.csv.gz"

sys.path.insert(0, str(OUT))
from build_digital_spending_analysis import (  # noqa: E402
    BROAD_ORDER,
    EXPENSE_PATHS,
    IPCA_INDEX,
    IPCA_TARGET,
    PELOTAS_CODE,
    base_expense_type,
    classify_broad,
    normalize_id,
    parse_money,
)


YEARS = [2016, 2020, 2024]
YEAR_COLORS = {2016: "#0072B2", 2020: "#E69F00", 2024: "#009E73"}


def load_pelotas_expense_lines() -> pd.DataFrame:
    parts = []
    for year, path in EXPENSE_PATHS.items():
        if year == 2016:
            columns = [
                "Sequencial Candidato",
                "Cargo",
                "Sigla da UE",
                "Valor despesa",
                "Tipo despesa",
            ]
            rename = {
                "Sequencial Candidato": "id_candidato",
                "Cargo": "cargo",
                "Sigla da UE": "codigo_municipio",
                "Valor despesa": "valor",
                "Tipo despesa": "tipo_despesa_bruto",
            }
        else:
            columns = [
                "SQ_CANDIDATO",
                "DS_CARGO",
                "SG_UE",
                "VR_DESPESA_CONTRATADA",
                "DS_ORIGEM_DESPESA",
            ]
            rename = {
                "SQ_CANDIDATO": "id_candidato",
                "DS_CARGO": "cargo",
                "SG_UE": "codigo_municipio",
                "VR_DESPESA_CONTRATADA": "valor",
                "DS_ORIGEM_DESPESA": "tipo_despesa_bruto",
            }

        for chunk in pd.read_csv(
            path,
            sep=";",
            encoding="latin1",
            dtype=str,
            usecols=columns,
            chunksize=200_000,
            low_memory=False,
        ):
            chunk = chunk.rename(columns=rename)
            chunk["codigo_municipio"] = normalize_id(
                chunk["codigo_municipio"]
            ).str.zfill(5)
            chunk = chunk[
                chunk["cargo"].fillna("").str.upper().eq("VEREADOR")
                & chunk["codigo_municipio"].eq(PELOTAS_CODE)
            ].copy()
            if chunk.empty:
                continue
            chunk["valor_nominal"] = parse_money(chunk["valor"], year)
            chunk = chunk[chunk["valor_nominal"].gt(0)].copy()
            chunk["id_candidato"] = normalize_id(chunk["id_candidato"])
            chunk["tipo_despesa"] = base_expense_type(
                chunk["tipo_despesa_bruto"]
            )
            chunk["grupo_despesa"] = chunk["tipo_despesa"].map(
                classify_broad
            )
            chunk["ano"] = year
            parts.append(
                chunk[
                    [
                        "ano",
                        "id_candidato",
                        "grupo_despesa",
                        "valor_nominal",
                    ]
                ]
            )
    return pd.concat(parts, ignore_index=True)


def candidate_matrix(lines: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        lines.groupby(
            ["ano", "id_candidato", "grupo_despesa"],
            as_index=False,
        )["valor_nominal"]
        .sum()
        .pivot(
            index=["ano", "id_candidato"],
            columns="grupo_despesa",
            values="valor_nominal",
        )
        .fillna(0)
        .reset_index()
    )
    pelotas = panel[
        panel["codigo_municipio"].str.zfill(5).eq(PELOTAS_CODE)
    ][
        [
            "ano",
            "id_candidato",
            "nome_candidato",
            "partido",
            "eleito",
            "receita_total_2024",
            "despesa_total_2024",
        ]
    ].copy()
    result = pelotas.merge(
        aggregate,
        on=["ano", "id_candidato"],
        how="left",
        validate="one_to_one",
    )
    for group in BROAD_ORDER:
        if group not in result:
            result[group] = 0.0
        result[group] = result[group].fillna(0)
        factor = result["ano"].map(
            lambda year: IPCA_TARGET / IPCA_INDEX[year]
        )
        result[f"{group} — R$ 2024"] = result[group] * factor
    return result


def summarize(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        year_frame = matrix[matrix["ano"].eq(year)]
        for group in BROAD_ORDER:
            column = f"{group} — R$ 2024"
            elected = year_frame[year_frame["eleito"].eq(1)]
            non_elected = year_frame[year_frame["eleito"].eq(0)]
            rows.append(
                {
                    "ano": year,
                    "grupo_despesa": group,
                    "valor_total_2024": year_frame[column].sum(),
                    "fracao_da_despesa_total": (
                        year_frame[column].sum()
                        / year_frame["despesa_total_2024"].sum()
                    ),
                    "auc_eleicao": roc_auc_score(
                        year_frame["eleito"], year_frame[column]
                    ),
                    "taxa_adocao_eleitos": elected[column].gt(0).mean(),
                    "taxa_adocao_nao_eleitos": non_elected[column].gt(0).mean(),
                    "mediana_eleitos_2024": elected[column].median(),
                    "mediana_nao_eleitos_2024": non_elected[column].median(),
                    "media_eleitos_2024": elected[column].mean(),
                    "media_nao_eleitos_2024": non_elected[column].mean(),
                }
            )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    ordered = (
        summary.groupby("grupo_despesa")["auc_eleicao"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for year, marker in zip(YEARS, ["o", "s", "^"]):
        frame = (
            summary[summary["ano"].eq(year)]
            .set_index("grupo_despesa")
            .reindex(ordered)
        )
        ax.scatter(
            frame["auc_eleicao"],
            y,
            s=56,
            marker=marker,
            color=YEAR_COLORS[year],
            label=str(year),
            zorder=3,
        )
        ax.plot(
            frame["auc_eleicao"],
            y,
            color=YEAR_COLORS[year],
            alpha=0.25,
            linewidth=0.8,
        )
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y, ordered)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC do gasto na categoria")
    ax.set_title(
        "Pelotas: quais despesas mais distinguem eleitos de não eleitos?",
        fontweight="bold",
    )
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.legend(title="Ano", frameon=False, ncol=3)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_8_auc_categorias_pelotas.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    panel = pd.read_csv(
        PANEL_PATH,
        dtype={"codigo_municipio": str, "id_candidato": str},
    )
    lines = load_pelotas_expense_lines()
    matrix = candidate_matrix(lines, panel)
    summary = summarize(matrix)
    matrix.to_csv(
        TABLE_OUT / "pelotas_categorias_despesa_candidato.csv",
        index=False,
    )
    summary.to_csv(
        TABLE_OUT / "pelotas_categorias_despesa_resumo.csv",
        index=False,
    )
    make_figure(summary)
    print(
        summary.sort_values(
            ["ano", "auc_eleicao"],
            ascending=[True, False],
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
