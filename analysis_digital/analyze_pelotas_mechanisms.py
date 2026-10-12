"""Diagnóstico de mecanismos alternativos para o acoplamento em Pelotas.

O objetivo não é atribuir causalidade, mas verificar quanto do padrão bruto
é compatível com: tamanho da disputa, longa cauda de candidaturas pouco
financiadas, concentração, seleção dentro dos partidos e incumbência.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mecanismos-pelotas")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TABLE_OUT = OUT / "tabelas"
FIGURE_OUT = OUT / "figuras"
PANEL_PATH = OUT / "dados" / "painel_candidatos_despesas_digitais.csv.gz"
CONTEXT_PATH = ROOT / "analysis_statewide" / "tabelas" / "contexto_municipal.csv"
YEARS = [2016, 2020, 2024]
PELOTAS_CODE = "87912"
YEAR_COLORS = {2016: "#0072B2", 2020: "#E69F00", 2024: "#009E73"}


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").upper())
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def safe_auc(y, score) -> float:
    y_array = np.asarray(y, dtype=int)
    return (
        float(roc_auc_score(y_array, np.asarray(score, dtype=float)))
        if len(np.unique(y_array)) > 1
        else np.nan
    )


def gini(values) -> float:
    array = np.sort(np.maximum(np.asarray(values, dtype=float), 0))
    if not len(array) or np.isclose(array.sum(), 0):
        return np.nan
    n = len(array)
    return (
        2 * np.sum(np.arange(1, n + 1) * array) / (n * array.sum())
        - (n + 1) / n
    )


def percentile_rank(series: pd.Series, value: float) -> float:
    valid = series.dropna()
    return float(
        (valid.lt(value).sum() + 0.5 * valid.eq(value).sum()) / len(valid)
    )


def stratified_auc(
    frame: pd.DataFrame,
    strata: list[str],
    score: str = "receita_total_2024",
) -> tuple[float, int, int]:
    """Concordância eleito–não eleito apenas dentro do mesmo estrato."""
    wins = 0
    ties = 0
    pairs = 0
    informative_groups = 0
    for _, group in frame.groupby(strata, sort=False):
        elected = group.loc[group["eleito"].eq(1), score].to_numpy(float)
        non_elected = group.loc[group["eleito"].eq(0), score].to_numpy(float)
        if not len(elected) or not len(non_elected):
            continue
        differences = elected[:, None] - non_elected[None, :]
        wins += int((differences > 0).sum())
        ties += int((differences == 0).sum())
        pairs += int(differences.size)
        informative_groups += 1
    value = (wins + 0.5 * ties) / pairs if pairs else np.nan
    return float(value), pairs, informative_groups


def add_incumbency(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["nome_normalizado"] = result["nome_candidato"].map(normalize_name)
    result["incumbente_nome_exato"] = 0
    for current_year, previous_year in [(2020, 2016), (2024, 2020)]:
        previous_winners = set(
            zip(
                result.loc[
                    result["ano"].eq(previous_year) & result["eleito"].eq(1),
                    "codigo_municipio",
                ],
                result.loc[
                    result["ano"].eq(previous_year) & result["eleito"].eq(1),
                    "nome_normalizado",
                ],
            )
        )
        current = result["ano"].eq(current_year)
        result.loc[current, "incumbente_nome_exato"] = [
            int(pair in previous_winners)
            for pair in zip(
                result.loc[current, "codigo_municipio"],
                result.loc[current, "nome_normalizado"],
            )
        ]
    return result


def city_year_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, code, municipality), group in panel.groupby(
        ["ano", "codigo_municipio", "municipio"],
        sort=True,
    ):
        candidates = len(group)
        elected = int(group["eleito"].sum())
        within_party, party_pairs, party_groups = stratified_auc(
            group, ["partido"]
        )
        rows.append(
            {
                "ano": year,
                "codigo_municipio": code,
                "municipio": municipality,
                "candidatos": candidates,
                "eleitos": elected,
                "candidatos_por_vaga": candidates / elected,
                "fracao_receita_zero": group["receita_total_2024"].eq(0).mean(),
                "gini_receita": gini(group["receita_total_2024"]),
                "auc_receita": safe_auc(
                    group["eleito"], group["receita_total_2024"]
                ),
                "auc_receita_dentro_partido": within_party,
                "pares_informativos_partido": party_pairs,
                "partidos_informativos": party_groups,
            }
        )
    result = pd.DataFrame(rows)

    for year in YEARS:
        mask = result["ano"].eq(year)
        year_frame = result.loc[mask]
        log_size = np.log1p(year_frame["candidatos"].to_numpy(float))
        design = np.column_stack([np.ones(len(log_size)), log_size, log_size**2])
        beta = np.linalg.lstsq(
            design, year_frame["auc_receita"].to_numpy(float), rcond=None
        )[0]
        expected = design @ beta
        residual = year_frame["auc_receita"].to_numpy(float) - expected
        result.loc[mask, "auc_esperada_pelo_numero_candidatos"] = expected
        result.loc[mask, "auc_residuo_numero_candidatos"] = residual
        result.loc[mask, "auc_percentil_ajustado_numero_candidatos"] = (
            pd.Series(residual, index=year_frame.index).rank(pct=True)
        )
        gini_values = year_frame["gini_receita"].to_numpy(float)
        gini_beta = np.linalg.lstsq(design, gini_values, rcond=None)[0]
        gini_expected = design @ gini_beta
        gini_residual = gini_values - gini_expected
        result.loc[mask, "gini_esperado_pelo_numero_candidatos"] = gini_expected
        result.loc[mask, "gini_residuo_numero_candidatos"] = gini_residual
        result.loc[mask, "gini_percentil_ajustado_numero_candidatos"] = (
            pd.Series(gini_residual, index=year_frame.index).rank(pct=True)
        )
    return result


def pelotas_mechanisms(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    quartile_rows = []
    incumbency_rows = []
    for year in YEARS:
        group = panel[
            panel["ano"].eq(year) & panel["codigo_municipio"].eq(PELOTAS_CODE)
        ].copy()
        city = metrics[
            metrics["ano"].eq(year)
            & metrics["codigo_municipio"].eq(PELOTAS_CODE)
        ].iloc[0]
        state_year = metrics[metrics["ano"].eq(year)]
        non_incumbents = group[group["incumbente_nome_exato"].eq(0)]
        top_decile_count = max(1, int(np.ceil(0.10 * len(group))))
        top_decile_share = (
            group.nlargest(top_decile_count, "receita_total_2024")[
                "receita_total_2024"
            ].sum()
            / group["receita_total_2024"].sum()
        )
        rows.append(
            {
                "ano": year,
                "candidatos": len(group),
                "vagas": int(group["eleito"].sum()),
                "candidatos_por_vaga": city["candidatos_por_vaga"],
                "percentil_rs_candidatos_por_vaga": percentile_rank(
                    state_year["candidatos_por_vaga"],
                    city["candidatos_por_vaga"],
                ),
                "fracao_receita_zero": city["fracao_receita_zero"],
                "percentil_rs_receita_zero": percentile_rank(
                    state_year["fracao_receita_zero"],
                    city["fracao_receita_zero"],
                ),
                "fracao_nao_eleitos_receita_zero": group.loc[
                    group["eleito"].eq(0), "receita_total_2024"
                ].eq(0).mean(),
                "fracao_eleitos_receita_zero": group.loc[
                    group["eleito"].eq(1), "receita_total_2024"
                ].eq(0).mean(),
                "gini_receita": city["gini_receita"],
                "percentil_rs_gini_receita": percentile_rank(
                    state_year["gini_receita"], city["gini_receita"]
                ),
                "gini_esperado_pelo_numero_candidatos": city[
                    "gini_esperado_pelo_numero_candidatos"
                ],
                "percentil_gini_ajustado_numero_candidatos": city[
                    "gini_percentil_ajustado_numero_candidatos"
                ],
                "participacao_top_10pct_candidatos": top_decile_share,
                "auc_receita_bruta": city["auc_receita"],
                "percentil_rs_auc_bruta": percentile_rank(
                    state_year["auc_receita"], city["auc_receita"]
                ),
                "auc_esperada_pelo_numero_candidatos": city[
                    "auc_esperada_pelo_numero_candidatos"
                ],
                "auc_residuo_numero_candidatos": city[
                    "auc_residuo_numero_candidatos"
                ],
                "percentil_ajustado_numero_candidatos": city[
                    "auc_percentil_ajustado_numero_candidatos"
                ],
                "auc_dentro_do_mesmo_partido": city[
                    "auc_receita_dentro_partido"
                ],
                "reducao_auc_ao_controlar_partido": (
                    city["auc_receita"]
                    - city["auc_receita_dentro_partido"]
                ),
                "incumbentes_identificados": int(
                    group["incumbente_nome_exato"].sum()
                ),
                "incumbentes_eleitos": int(
                    group.loc[
                        group["incumbente_nome_exato"].eq(1), "eleito"
                    ].sum()
                ),
                "auc_receita_sem_incumbentes": (
                    safe_auc(
                        non_incumbents["eleito"],
                        non_incumbents["receita_total_2024"],
                    )
                    if year >= 2020
                    else np.nan
                ),
            }
        )

        quartiles = pd.qcut(
            group["receita_total_2024"].rank(method="first"),
            4,
            labels=["Q1: menor", "Q2", "Q3", "Q4: maior"],
        )
        for quartile, subset in group.groupby(quartiles, observed=True):
            quartile_rows.append(
                {
                    "ano": year,
                    "quartil_receita": quartile,
                    "candidatos": len(subset),
                    "eleitos": int(subset["eleito"].sum()),
                    "taxa_eleicao": subset["eleito"].mean(),
                    "receita_mediana_2024": subset[
                        "receita_total_2024"
                    ].median(),
                }
            )

        if year >= 2020:
            for incumbent, label in [
                (0, "Demais candidaturas"),
                (1, "Incumbentes identificados"),
            ]:
                subset = group[group["incumbente_nome_exato"].eq(incumbent)]
                incumbency_rows.append(
                    {
                        "ano": year,
                        "grupo": label,
                        "candidatos": len(subset),
                        "eleitos": int(subset["eleito"].sum()),
                        "taxa_eleicao": subset["eleito"].mean(),
                        "receita_mediana_2024": subset[
                            "receita_total_2024"
                        ].median(),
                    }
                )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(quartile_rows),
        pd.DataFrame(incumbency_rows),
    )


def correlations(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        frame = metrics[metrics["ano"].eq(year)]
        for predictor in [
            "candidatos",
            "candidatos_por_vaga",
            "fracao_receita_zero",
            "gini_receita",
            "auc_receita_dentro_partido",
        ]:
            valid = frame[["auc_receita", predictor]].dropna()
            rho, p_value = spearmanr(valid["auc_receita"], valid[predictor])
            rows.append(
                {
                    "ano": year,
                    "desfecho": "auc_receita",
                    "preditor": predictor,
                    "rho_spearman": rho,
                    "p_valor": p_value,
                    "municipios": len(valid),
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    mechanisms: pd.DataFrame,
    quartiles: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(YEARS))
    width = 0.25
    series = [
        ("AUC observada", "auc_receita_bruta", "#6A3D9A"),
        (
            "AUC esperada pelo nº de candidatos",
            "auc_esperada_pelo_numero_candidatos",
            "#8A98A8",
        ),
        (
            "AUC dentro do mesmo partido",
            "auc_dentro_do_mesmo_partido",
            "#009E73",
        ),
    ]
    for index, (label, column, color) in enumerate(series):
        ax.bar(
            x + (index - 1) * width,
            mechanisms[column],
            width=width,
            label=label,
            color=color,
        )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x, YEARS)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUC receita–eleição")
    ax.set_title(
        "Pelotas: decomposição estrutural do acoplamento",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_6_mecanismos_pelotas.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    labels = ["Q1: menor", "Q2", "Q3", "Q4: maior"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.7, 4.5))
    for index, year in enumerate(YEARS):
        frame = (
            quartiles[quartiles["ano"].eq(year)]
            .set_index("quartil_receita")
            .reindex(labels)
        )
        ax.bar(
            x + (index - 1) * width,
            100 * frame["taxa_eleicao"],
            width=width,
            label=str(year),
            color=YEAR_COLORS[year],
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("% eleito no quartil")
    ax.set_title(
        "Pelotas: a eleição se concentra no quartil mais financiado",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.legend(title="Ano", frameon=False, ncol=3)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_7_quartis_receita_pelotas.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    panel = pd.read_csv(
        PANEL_PATH,
        dtype={"codigo_municipio": str, "id_candidato": str},
    )
    panel["codigo_municipio"] = panel["codigo_municipio"].str.zfill(5)
    panel = add_incumbency(panel)
    metrics = city_year_metrics(panel)
    mechanisms, quartiles, incumbency = pelotas_mechanisms(panel, metrics)
    correlation_table = correlations(metrics)

    metrics.to_csv(
        TABLE_OUT / "mecanismos_indicadores_municipio_ano.csv",
        index=False,
    )
    mechanisms.to_csv(
        TABLE_OUT / "mecanismos_pelotas.csv",
        index=False,
    )
    quartiles.to_csv(
        TABLE_OUT / "pelotas_quartis_receita.csv",
        index=False,
    )
    incumbency.to_csv(
        TABLE_OUT / "pelotas_incumbencia.csv",
        index=False,
    )
    correlation_table.to_csv(
        TABLE_OUT / "correlacoes_mecanismos_rs.csv",
        index=False,
    )
    make_figures(mechanisms, quartiles)
    print(mechanisms.to_string(index=False))


if __name__ == "__main__":
    main()
