"""Contexto territorial e tipologia de exceções ao padrão estadual.

O script combina os indicadores eleitorais já estimados com divisões regionais
do IBGE, população e urbanização do Censo 2022, PIB per capita de 2023 e IDHM
municipal de 2010. A classificação de exceções é descritiva e pré-especificada
por regras transparentes; não implica causalidade.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, spearmanr


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tabelas"
EXTERNAL = ROOT / "dados_externos"
YEARS = [2016, 2020, 2024]


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.upper().replace("’", "'")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return adjusted
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted.loc[order] = np.minimum(monotone, 1.0)
    return adjusted


def read_aggregate_series(path: Path, year: str, value_name: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for result in payload[0]["resultados"]:
        for item in result["series"]:
            code = str(item["localidade"]["id"])
            if code.startswith("43"):
                value = pd.to_numeric(item["serie"].get(year), errors="coerce")
                rows.append({"codigo_ibge": code, value_name: value})
    return pd.DataFrame(rows)


def read_pib_per_capita() -> pd.DataFrame:
    payload = json.loads(
        (EXTERNAL / "ibge_pib_per_capita_2023.json").read_text(encoding="utf-8")
    )
    rows = []
    for item in payload[0]["res"]:
        rows.append(
            {
                "codigo_ibge_6": str(item["localidade"]),
                "pib_per_capita_2023": pd.to_numeric(
                    item["res"].get("2023"), errors="coerce"
                ),
            }
        )
    return pd.DataFrame(rows)


def read_geography() -> pd.DataFrame:
    payload = json.loads(
        (EXTERNAL / "ibge_localidades_rs.json").read_text(encoding="utf-8")
    )
    rows = []
    for item in payload:
        immediate = item["regiao-imediata"]
        intermediate = immediate["regiao-intermediaria"]
        micro = item["microrregiao"]
        meso = micro["mesorregiao"]
        rows.append(
            {
                "codigo_ibge": str(item["id"]),
                "codigo_ibge_6": str(item["id"])[:6],
                "municipio_ibge": item["nome"],
                "chave_municipio": normalize_name(item["nome"]),
                "regiao_imediata": immediate["nome"],
                "regiao_intermediaria": intermediate["nome"],
                "microrregiao_1990": micro["nome"],
                "mesorregiao_1990": meso["nome"],
            }
        )
    return pd.DataFrame(rows)


def read_idhm() -> pd.DataFrame:
    path = EXTERNAL / "atlas2013_dadosbrutos_pt.csv"
    columns = [
        "ANO",
        "UF",
        "Codmun7",
        "Município",
        "IDHM",
        "IDHM_E",
        "IDHM_L",
        "IDHM_R",
        "RDPC",
        "GINI",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame[
        frame["ANO"].eq(2010) & frame["UF"].eq("RIO GRANDE DO SUL")
    ].copy()
    frame["codigo_ibge"] = frame["Codmun7"].astype("Int64").astype(str)
    frame["chave_municipio"] = frame["Município"].map(normalize_name)
    frame = frame.rename(
        columns={
            "IDHM": "idhm_2010",
            "IDHM_E": "idhm_educacao_2010",
            "IDHM_L": "idhm_longevidade_2010",
            "IDHM_R": "idhm_renda_2010",
            "RDPC": "renda_per_capita_2010",
            "GINI": "gini_renda_2010",
        }
    )
    return frame[
        [
            "codigo_ibge",
            "chave_municipio",
            "idhm_2010",
            "idhm_educacao_2010",
            "idhm_longevidade_2010",
            "idhm_renda_2010",
            "renda_per_capita_2010",
            "gini_renda_2010",
        ]
    ]


def add_adjusted_percentiles(
    context: pd.DataFrame,
    metric_prefix: str,
    candidate_prefix: str,
    years: list[int],
) -> pd.DataFrame:
    for year in years:
        metric = f"{metric_prefix}_{year}"
        candidate_count = f"{candidate_prefix}_{year}"
        log_n = np.log1p(context[candidate_count].astype(float))
        design = np.column_stack([np.ones(len(context)), log_n, log_n**2])
        coefficients = np.linalg.lstsq(
            design, context[metric].astype(float), rcond=None
        )[0]
        residual = context[metric].astype(float) - design @ coefficients
        context[f"{metric_prefix}_residuo_tamanho_{year}"] = residual
        context[f"{metric_prefix}_percentil_ajustado_{year}"] = residual.rank(
            method="average", pct=True
        )
    return context


def build_context() -> tuple[pd.DataFrame, dict]:
    changes = pd.read_csv(TABLES / "mudancas_por_municipio.csv")
    changes["chave_municipio"] = changes["municipio"].map(normalize_name)

    geography = read_geography()
    population = read_aggregate_series(
        EXTERNAL / "ibge_populacao_2022.json", "2022", "populacao_2022"
    )
    urbanization = read_aggregate_series(
        EXTERNAL / "ibge_urbanizacao_2022.json", "2022", "urbanizacao_2022"
    )
    pib = read_pib_per_capita()
    idhm = read_idhm()

    external = (
        geography.merge(population, on="codigo_ibge", how="left", validate="1:1")
        .merge(urbanization, on="codigo_ibge", how="left", validate="1:1")
        .merge(pib, on="codigo_ibge_6", how="left", validate="1:1")
        .merge(
            idhm.drop(columns="chave_municipio"),
            on="codigo_ibge",
            how="left",
            validate="1:1",
        )
    )
    context = changes.merge(
        external,
        on="chave_municipio",
        how="left",
        validate="1:1",
        indicator=True,
    )
    if not context["_merge"].eq("both").all():
        missing = context.loc[
            ~context["_merge"].eq("both"), ["municipio", "chave_municipio"]
        ]
        raise ValueError(f"Municípios sem correspondência IBGE:\n{missing}")
    context = context.drop(columns="_merge")

    context["log10_populacao_2022"] = np.log10(context["populacao_2022"])
    context["log10_pib_per_capita_2023"] = np.log10(
        context["pib_per_capita_2023"]
    )
    context["faixa_populacao_2022"] = pd.cut(
        context["populacao_2022"],
        bins=[0, 10_000, 20_000, 50_000, 100_000, np.inf],
        labels=[
            "Até 10 mil",
            "10–20 mil",
            "20–50 mil",
            "50–100 mil",
            "Mais de 100 mil",
        ],
        include_lowest=True,
        right=True,
    )
    context["quartil_idhm_2010"] = pd.qcut(
        context["idhm_2010"],
        q=4,
        labels=["Q1: menor", "Q2", "Q3", "Q4: maior"],
        duplicates="drop",
    )
    context["quartil_pib_pc_2023"] = pd.qcut(
        context["pib_per_capita_2023"],
        q=4,
        labels=["Q1: menor", "Q2", "Q3", "Q4: maior"],
        duplicates="drop",
    )
    context["quartil_urbanizacao_2022"] = pd.qcut(
        context["urbanizacao_2022"],
        q=4,
        labels=["Q1: menor", "Q2", "Q3", "Q4: maior"],
        duplicates="drop",
    )

    context = add_adjusted_percentiles(
        context, "auc_receita_eleicao", "candidatos", YEARS
    )
    context = add_adjusted_percentiles(
        context, "gini_receita", "candidatos", [2024]
    )

    raw_auc_thresholds = {
        year: float(context[f"auc_receita_eleicao_{year}"].quantile(0.90))
        for year in YEARS
    }
    for year in YEARS:
        context[f"auc_top10_bruto_{year}"] = context[
            f"auc_receita_eleicao_{year}"
        ].ge(raw_auc_thresholds[year])
        context[f"auc_top10_ajustado_{year}"] = context[
            f"auc_receita_eleicao_percentil_ajustado_{year}"
        ].ge(0.90)

    context["acoplamento_alto_persistente_bruto"] = context[
        [f"auc_top10_bruto_{year}" for year in YEARS]
    ].sum(axis=1).ge(2)
    context["acoplamento_alto_persistente_ajustado"] = context[
        [f"auc_top10_ajustado_{year}" for year in YEARS]
    ].sum(axis=1).ge(2)
    context["acoplamento_alto_persistente"] = context[
        [
            "acoplamento_alto_persistente_bruto",
            "acoplamento_alto_persistente_ajustado",
        ]
    ].any(axis=1)
    context["contratendencia_auc"] = context[
        "variacao_auc_receita_eleicao_2024_menos_2016"
    ].ge(0)
    context["contratendencia_auc_forte"] = context[
        "variacao_auc_receita_eleicao_2024_menos_2016"
    ].ge(0.10)
    context["concentracao_alta_2024_ajustada"] = context[
        "gini_receita_percentil_ajustado_2024"
    ].ge(0.90)
    public_threshold = float(context["fracao_publica_2024"].quantile(0.10))
    context["publicizacao_baixa_2024"] = context["fracao_publica_2024"].le(
        public_threshold
    )
    exception_columns = [
        "acoplamento_alto_persistente",
        "contratendencia_auc_forte",
        "concentracao_alta_2024_ajustada",
        "publicizacao_baixa_2024",
    ]
    context["escore_excecao"] = context[exception_columns].sum(axis=1)
    context["excecao_multidimensional"] = context["escore_excecao"].ge(2)
    context["tipos_excecao"] = context.apply(
        lambda row: "; ".join(
            label
            for column, label in [
                ("acoplamento_alto_persistente", "acoplamento persistente"),
                ("contratendencia_auc_forte", "forte contratendência da AUC"),
                (
                    "concentracao_alta_2024_ajustada",
                    "concentração elevada em 2024",
                ),
                ("publicizacao_baixa_2024", "baixa publicização em 2024"),
            ]
            if row[column]
        ),
        axis=1,
    )

    thresholds = {
        "auc_top10_bruto": raw_auc_thresholds,
        "auc_top10_ajustado": "percentil >= 0,90 do resíduo de AUC após ajuste quadrático por log(1 + número de candidatos), em cada ano",
        "acoplamento_alto_persistente": "top 10% bruto ou ajustado por tamanho em pelo menos dois dos três anos",
        "contratendencia_auc": "AUC 2024 - AUC 2016 >= 0",
        "contratendencia_auc_forte": "AUC 2024 - AUC 2016 >= 0,10",
        "concentracao_alta_2024_ajustada": "percentil >= 0,90 do resíduo de Gini após ajuste quadrático por log(1 + número de candidatos)",
        "publicizacao_baixa_2024": f"fração pública <= percentil 10 estadual ({public_threshold:.6f})",
        "excecao_multidimensional": "dois ou mais dos quatro critérios principais",
        "idhm": "Atlas do Desenvolvimento Humano, Censo 2010; Pinto Bandeira sem valor porque foi instalado após o censo",
    }
    return context, thresholds


def summarize_groups(context: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_name, group in context.groupby(group_column, observed=True):
        rows.append(
            {
                group_column: group_name,
                "municipios": len(group),
                "mediana_delta_auc_2016_2024": group[
                    "variacao_auc_receita_eleicao_2024_menos_2016"
                ].median(),
                "mediana_auc_2024": group["auc_receita_eleicao_2024"].median(),
                "mediana_gini_2024": group["gini_receita_2024"].median(),
                "mediana_fracao_publica_2024": group[
                    "fracao_publica_2024"
                ].median(),
                "percentual_contratendencia_auc": 100
                * group["contratendencia_auc"].mean(),
                "percentual_contratendencia_auc_forte": 100
                * group["contratendencia_auc_forte"].mean(),
                "percentual_acoplamento_alto_persistente": 100
                * group["acoplamento_alto_persistente"].mean(),
                "percentual_excecao_multidimensional": 100
                * group["excecao_multidimensional"].mean(),
            }
        )
    return pd.DataFrame(rows)


def contextual_tests(context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    covariates = {
        "IDHM 2010": "idhm_2010",
        "População 2022 (log10)": "log10_populacao_2022",
        "Urbanização 2022": "urbanizacao_2022",
        "PIB per capita 2023 (log10)": "log10_pib_per_capita_2023",
    }
    outcomes = {
        "Variação da AUC, 2016–2024": "variacao_auc_receita_eleicao_2024_menos_2016",
        "AUC em 2024": "auc_receita_eleicao_2024",
        "Variação do Gini, 2016–2024": "variacao_gini_receita_2024_menos_2016",
        "Fração pública em 2024": "fracao_publica_2024",
    }
    for covariate_label, covariate in covariates.items():
        for outcome_label, outcome in outcomes.items():
            valid = context[[covariate, outcome]].dropna()
            rho, p_value = spearmanr(valid[covariate], valid[outcome])
            rows.append(
                {
                    "familia": "Correlação de Spearman",
                    "relacao": f"{covariate_label} × {outcome_label}",
                    "n": len(valid),
                    "estatistica": rho,
                    "graus_liberdade": np.nan,
                    "p_valor": p_value,
                }
            )

    for flag in [
        "contratendencia_auc",
        "contratendencia_auc_forte",
        "acoplamento_alto_persistente",
        "excecao_multidimensional",
    ]:
        table = pd.crosstab(context["regiao_intermediaria"], context[flag])
        statistic, p_value, degrees, _ = chi2_contingency(table)
        rows.append(
            {
                "familia": "Qui-quadrado por região intermediária",
                "relacao": flag,
                "n": len(context),
                "estatistica": statistic,
                "graus_liberdade": degrees,
                "p_valor": p_value,
            }
        )

    regional_samples = [
        group["variacao_auc_receita_eleicao_2024_menos_2016"].dropna().to_numpy()
        for _, group in context.groupby("regiao_intermediaria")
    ]
    statistic, p_value = kruskal(*regional_samples)
    rows.append(
        {
            "familia": "Kruskal–Wallis por região intermediária",
            "relacao": "Variação da AUC, 2016–2024",
            "n": len(context),
            "estatistica": statistic,
            "graus_liberdade": len(regional_samples) - 1,
            "p_valor": p_value,
        }
    )
    tests = pd.DataFrame(rows)
    tests["p_bh"] = benjamini_hochberg(tests["p_valor"])
    return tests


def main() -> None:
    context, thresholds = build_context()
    context = context.sort_values(
        [
            "escore_excecao",
            "acoplamento_alto_persistente",
            "variacao_auc_receita_eleicao_2024_menos_2016",
        ],
        ascending=[False, False, False],
    )
    highlights = context[
        context["excecao_multidimensional"]
        | context["acoplamento_alto_persistente"]
        | context["contratendencia_auc_forte"]
        | context["municipio"].eq("PELOTAS")
    ].copy()

    context.to_csv(TABLES / "contexto_municipal.csv", index=False)
    highlights.to_csv(TABLES / "cidades_fora_padrao.csv", index=False)
    summarize_groups(context, "regiao_intermediaria").to_csv(
        TABLES / "excecoes_por_regiao_intermediaria.csv", index=False
    )
    summarize_groups(context, "quartil_idhm_2010").to_csv(
        TABLES / "excecoes_por_quartil_idhm.csv", index=False
    )
    summarize_groups(context, "faixa_populacao_2022").to_csv(
        TABLES / "excecoes_por_faixa_populacao.csv", index=False
    )
    contextual_tests(context).to_csv(
        TABLES / "testes_contexto_municipal.csv", index=False
    )
    (TABLES / "criterios_excecoes_municipais.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "municipios": int(len(context)),
        "idhm_disponivel": int(context["idhm_2010"].notna().sum()),
        "contratendencia_auc": int(context["contratendencia_auc"].sum()),
        "contratendencia_auc_forte": int(
            context["contratendencia_auc_forte"].sum()
        ),
        "acoplamento_alto_persistente": int(
            context["acoplamento_alto_persistente"].sum()
        ),
        "concentracao_alta_2024_ajustada": int(
            context["concentracao_alta_2024_ajustada"].sum()
        ),
        "publicizacao_baixa_2024": int(
            context["publicizacao_baixa_2024"].sum()
        ),
        "excecao_multidimensional": int(
            context["excecao_multidimensional"].sum()
        ),
        "destaques_exportados": int(len(highlights)),
    }
    (TABLES / "resumo_excecoes_municipais.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
