"""Despesas digitais nas campanhas a vereador do Rio Grande do Sul.

O script lê os arquivos oficiais de despesas contratadas/finais do TSE,
harmoniza as rubricas de 2016, 2020 e 2024, integra-as ao painel de
candidaturas válidas e testa se a composição digital ajuda a explicar a
associação entre recursos e eleição, com ênfase no caso de Pelotas.

Valores reais são expressos em reais de outubro de 2024 pelos mesmos fatores
de IPCA usados no painel estadual de receitas.
"""

from __future__ import annotations

import json
import math
import os
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-despesas-digitais")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, mannwhitneyu, norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RAW = OUT / "dados_brutos"
DATA_OUT = OUT / "dados"
TABLE_OUT = OUT / "tabelas"
FIGURE_OUT = OUT / "figuras"
for directory in (DATA_OUT, TABLE_OUT, FIGURE_OUT):
    directory.mkdir(parents=True, exist_ok=True)

PANEL_PATH = ROOT / "analysis_statewide" / "dados" / "painel_candidatos_rs.csv.gz"
CONTEXT_PATH = ROOT / "analysis_statewide" / "tabelas" / "contexto_municipal.csv"
STATE_INDICATORS_PATH = (
    ROOT / "analysis_statewide" / "tabelas" / "indicadores_municipio_ano.csv"
)

sys.path.insert(0, str(ROOT / "analysis_statewide"))
from build_statewide_panel import (  # noqa: E402
    IPCA_INDEX,
    IPCA_TARGET,
    fit_conditional_logit,
)


YEARS = [2016, 2020, 2024]
PELOTAS_CODE = "87912"
EXPENSE_PATHS = {
    2016: RAW / "despesas_candidatos_prestacao_contas_final_2016_RS.txt",
    2020: RAW / "despesas_contratadas_candidatos_2020_RS.csv",
    2024: RAW / "despesas_contratadas_candidatos_2024_RS.csv",
}

DIGITAL_TYPES = {
    "Criação e inclusão de páginas na internet",
    "Despesa com Impulsionamento de Conteúdos",
}
AUDIOVISUAL_TYPES = {
    "Produção de jingles, vinhetas e slogans",
    "Produção de programas de rádio, televisão ou vídeo",
}
TRADITIONAL_TYPES = {
    "Publicidade por materiais impressos",
    "Publicidade por adesivos",
    "Publicidade por jornais e revistas",
    "Publicidade por carros de som",
    "Correspondências e despesas postais",
}
MOBILIZATION_TYPES = {
    "Atividades de militância e mobilização de rua",
    "Despesas com pessoal",
    "Encargos sociais",
    "Alimentação",
    "Comícios",
    "Eventos de promoção da candidatura",
}
PROFESSIONAL_TYPES = {
    "Serviços contábeis",
    "Serviços advocatícios",
    "Serviços prestados por terceiros",
    "Serviços próprios prestados por terceiros",
    "Pesquisas ou testes eleitorais",
}
TRANSPORT_TYPES = {
    "Combustíveis e lubrificantes",
    "Cessão ou locação de veículos",
    "Despesas com transporte ou deslocamento",
    "Despesas com Hospedagem",
}
ADMIN_TYPES = {
    "Encargos financeiros, taxas bancárias e/ou op. cartão de crédito",
    "Materiais de expediente",
    "Locação/cessão de bens imóveis",
    "Locação/cessão de bens móveis (exceto veículos)",
    "Energia elétrica",
    "Água",
    "Telefone",
    "Pré-instalação física de comitê de campanha",
    "Despesa com geradores de energia",
    "Impostos, contribuições e taxas",
    "Taxa de Administração de Financiamento Coletivo",
}

BROAD_ORDER = [
    "Digital estrito",
    "Audiovisual",
    "Publicidade tradicional",
    "Mobilização e pessoal",
    "Serviços profissionais",
    "Transporte e logística",
    "Infraestrutura e administração",
    "Transferências e outros",
]

COLORS = {
    2016: "#0072B2",
    2020: "#E69F00",
    2024: "#009E73",
    "Pelotas": "#6A3D9A",
    "RS": "#666666",
    "Eleitos": "#0072B2",
    "Não eleitos": "#B3B3B3",
}


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(character for character in text if not unicodedata.combining(character))


def parse_money(series: pd.Series, year: int) -> pd.Series:
    text = series.fillna("0").astype(str).str.strip()
    if year >= 2020:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").fillna(0.0)


def base_expense_type(series: pd.Series) -> pd.Series:
    return (
        series.fillna("#NULO")
        .astype(str)
        .str.strip()
        .str.replace("Baixa de Estimaveis - ", "", regex=False)
    )


def classify_broad(expense_type: str) -> str:
    if expense_type in DIGITAL_TYPES:
        return "Digital estrito"
    if expense_type in AUDIOVISUAL_TYPES:
        return "Audiovisual"
    if expense_type in TRADITIONAL_TYPES:
        return "Publicidade tradicional"
    if expense_type in MOBILIZATION_TYPES:
        return "Mobilização e pessoal"
    if expense_type in PROFESSIONAL_TYPES:
        return "Serviços profissionais"
    if expense_type in TRANSPORT_TYPES:
        return "Transporte e logística"
    if expense_type in ADMIN_TYPES:
        return "Infraestrutura e administração"
    return "Transferências e outros"


def clean_supplier(series: pd.Series) -> pd.Series:
    result = series.fillna("").astype(str).str.strip()
    result = result.mask(result.str.startswith("#NULO"), "")
    return result


def load_expenses(
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Lê um ano em blocos e retorna agregações, fornecedores e auditoria."""
    path = EXPENSE_PATHS[year]
    if year == 2016:
        columns = [
            "Sequencial Candidato",
            "Cargo",
            "Sigla da UE",
            "Nome da UE",
            "Sigla  Partido",
            "Número candidato",
            "Nome candidato",
            "CPF/CNPJ do fornecedor",
            "Nome do fornecedor",
            "Nome do fornecedor (Receita Federal)",
            "Data da despesa",
            "Valor despesa",
            "Tipo despesa",
            "Descriçao da despesa",
        ]
        rename = {
            "Sequencial Candidato": "id_candidato",
            "Cargo": "cargo",
            "Sigla da UE": "codigo_municipio",
            "Nome da UE": "municipio",
            "Sigla  Partido": "partido",
            "Número candidato": "numero_candidato",
            "Nome candidato": "nome_candidato",
            "CPF/CNPJ do fornecedor": "documento_fornecedor",
            "Nome do fornecedor": "fornecedor_declarado",
            "Nome do fornecedor (Receita Federal)": "fornecedor_rfb",
            "Data da despesa": "data_despesa",
            "Valor despesa": "valor",
            "Tipo despesa": "tipo_despesa_bruto",
            "Descriçao da despesa": "descricao",
        }
    else:
        columns = [
            "SQ_CANDIDATO",
            "DS_CARGO",
            "SG_UE",
            "NM_UE",
            "SG_PARTIDO",
            "NR_CANDIDATO",
            "NM_CANDIDATO",
            "NR_CPF_CNPJ_FORNECEDOR",
            "NM_FORNECEDOR",
            "NM_FORNECEDOR_RFB",
            "DT_DESPESA",
            "VR_DESPESA_CONTRATADA",
            "DS_ORIGEM_DESPESA",
            "DS_DESPESA",
        ]
        rename = {
            "SQ_CANDIDATO": "id_candidato",
            "DS_CARGO": "cargo",
            "SG_UE": "codigo_municipio",
            "NM_UE": "municipio",
            "SG_PARTIDO": "partido",
            "NR_CANDIDATO": "numero_candidato",
            "NM_CANDIDATO": "nome_candidato",
            "NR_CPF_CNPJ_FORNECEDOR": "documento_fornecedor",
            "NM_FORNECEDOR": "fornecedor_declarado",
            "NM_FORNECEDOR_RFB": "fornecedor_rfb",
            "DT_DESPESA": "data_despesa",
            "VR_DESPESA_CONTRATADA": "valor",
            "DS_ORIGEM_DESPESA": "tipo_despesa_bruto",
            "DS_DESPESA": "descricao",
        }

    candidate_parts: list[pd.DataFrame] = []
    type_parts: list[pd.DataFrame] = []
    supplier_parts: list[pd.DataFrame] = []
    rows_all = 0
    rows_council = 0
    rows_zero = 0
    rows_negative = 0
    candidates_all: set[str] = set()

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        dtype=str,
        usecols=columns,
        chunksize=100_000,
        low_memory=False,
    ):
        rows_all += len(chunk)
        chunk = chunk.rename(columns=rename)
        chunk = chunk[chunk["cargo"].fillna("").str.upper().eq("VEREADOR")].copy()
        rows_council += len(chunk)
        if chunk.empty:
            continue

        chunk["ano"] = year
        chunk["id_candidato"] = normalize_id(chunk["id_candidato"])
        chunk["codigo_municipio"] = normalize_id(chunk["codigo_municipio"]).str.zfill(5)
        chunk["valor_nominal"] = parse_money(chunk["valor"], year)
        rows_zero += int(chunk["valor_nominal"].eq(0).sum())
        rows_negative += int(chunk["valor_nominal"].lt(0).sum())
        chunk = chunk[chunk["valor_nominal"].gt(0)].copy()
        if chunk.empty:
            continue
        candidates_all.update(chunk["id_candidato"].unique())

        chunk["tipo_despesa"] = base_expense_type(chunk["tipo_despesa_bruto"])
        chunk["grupo_despesa"] = chunk["tipo_despesa"].map(classify_broad)
        chunk["estimavel_2016"] = (
            chunk["tipo_despesa_bruto"]
            .fillna("")
            .str.startswith("Baixa de Estimaveis - ")
            .astype(int)
        )
        chunk["digital"] = chunk["tipo_despesa"].isin(DIGITAL_TYPES).astype(int)
        chunk["impulsionamento"] = (
            chunk["tipo_despesa"].eq("Despesa com Impulsionamento de Conteúdos").astype(int)
        )
        chunk["paginas_internet"] = (
            chunk["tipo_despesa"].eq("Criação e inclusão de páginas na internet").astype(int)
        )
        chunk["audiovisual"] = chunk["tipo_despesa"].isin(AUDIOVISUAL_TYPES).astype(int)
        for indicator in ("digital", "impulsionamento", "paginas_internet", "audiovisual"):
            chunk[f"valor_{indicator}"] = chunk["valor_nominal"] * chunk[indicator]

        candidate_parts.append(
            chunk.groupby(
                ["ano", "id_candidato", "codigo_municipio", "municipio"],
                as_index=False,
            ).agg(
                despesa_total_nominal=("valor_nominal", "sum"),
                despesa_digital_nominal=("valor_digital", "sum"),
                despesa_impulsionamento_nominal=("valor_impulsionamento", "sum"),
                despesa_paginas_internet_nominal=("valor_paginas_internet", "sum"),
                despesa_audiovisual_nominal=("valor_audiovisual", "sum"),
                linhas_despesa=("valor_nominal", "size"),
            )
        )
        type_parts.append(
            chunk.groupby(
                ["ano", "tipo_despesa", "grupo_despesa"],
                as_index=False,
            ).agg(
                valor_nominal=("valor_nominal", "sum"),
                linhas=("valor_nominal", "size"),
                candidatos=("id_candidato", "nunique"),
            )
        )

        digital_pelotas = chunk[
            chunk["digital"].eq(1) & chunk["codigo_municipio"].eq(PELOTAS_CODE)
        ].copy()
        if not digital_pelotas.empty:
            digital_pelotas["fornecedor_rfb"] = clean_supplier(
                digital_pelotas["fornecedor_rfb"]
            )
            digital_pelotas["fornecedor_declarado"] = clean_supplier(
                digital_pelotas["fornecedor_declarado"]
            )
            digital_pelotas["fornecedor"] = digital_pelotas[
                "fornecedor_rfb"
            ].mask(
                digital_pelotas["fornecedor_rfb"].eq(""),
                digital_pelotas["fornecedor_declarado"],
            )
            digital_pelotas["fornecedor"] = digital_pelotas["fornecedor"].mask(
                digital_pelotas["fornecedor"].eq(""),
                "Fornecedor não identificado",
            )
            supplier_parts.append(
                digital_pelotas[
                    [
                        "ano",
                        "id_candidato",
                        "nome_candidato",
                        "partido",
                        "tipo_despesa",
                        "data_despesa",
                        "fornecedor",
                        "valor_nominal",
                    ]
                ]
            )

    candidate = pd.concat(candidate_parts, ignore_index=True)
    candidate = (
        candidate.groupby(
            ["ano", "id_candidato", "codigo_municipio"],
            as_index=False,
        )
        .agg(
            municipio_despesa=("municipio", "first"),
            despesa_total_nominal=("despesa_total_nominal", "sum"),
            despesa_digital_nominal=("despesa_digital_nominal", "sum"),
            despesa_impulsionamento_nominal=("despesa_impulsionamento_nominal", "sum"),
            despesa_paginas_internet_nominal=("despesa_paginas_internet_nominal", "sum"),
            despesa_audiovisual_nominal=("despesa_audiovisual_nominal", "sum"),
            linhas_despesa=("linhas_despesa", "sum"),
        )
    )
    expense_types = pd.concat(type_parts, ignore_index=True)
    expense_types = (
        expense_types.groupby(
            ["ano", "tipo_despesa", "grupo_despesa"],
            as_index=False,
        )
        .agg(
            valor_nominal=("valor_nominal", "sum"),
            linhas=("linhas", "sum"),
            candidatos=("candidatos", "sum"),
        )
    )
    suppliers = (
        pd.concat(supplier_parts, ignore_index=True)
        if supplier_parts
        else pd.DataFrame(
            columns=[
                "ano",
                "id_candidato",
                "nome_candidato",
                "partido",
                "tipo_despesa",
                "data_despesa",
                "fornecedor",
                "valor_nominal",
            ]
        )
    )
    audit = {
        "ano": year,
        "arquivo": path.name,
        "linhas_arquivo": rows_all,
        "linhas_vereador": rows_council,
        "linhas_valor_zero": rows_zero,
        "linhas_valor_negativo": rows_negative,
        "candidatos_com_despesa_positiva": len(candidates_all),
        "valor_total_nominal_vereadores": float(candidate["despesa_total_nominal"].sum()),
    }
    return candidate, expense_types, suppliers, audit


def safe_auc(y: pd.Series | np.ndarray, score: pd.Series | np.ndarray) -> float:
    y_array = np.asarray(y, dtype=int)
    if len(np.unique(y_array)) < 2:
        return np.nan
    return float(roc_auc_score(y_array, np.asarray(score, dtype=float)))


def percentile_rank(series: pd.Series, value: float) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty or pd.isna(value):
        return np.nan
    return float((valid.le(value).sum() - 0.5 * valid.eq(value).sum()) / len(valid))


def coefficient_rows(model: dict, model_name: str, year: int) -> pd.DataFrame:
    rows = []
    for parameter, beta, se in zip(
        model["parameter_names"],
        model["beta"],
        model["standard_error"],
    ):
        z_score = beta / se if se > 0 else np.nan
        rows.append(
            {
                "ano": year,
                "modelo": model_name,
                "parametro": parameter,
                "beta": beta,
                "erro_padrao_cluster": se,
                "odds_ratio": math.exp(beta),
                "or_ic95_inferior": math.exp(beta - 1.96 * se),
                "or_ic95_superior": math.exp(beta + 1.96 * se),
                "p_valor": 2 * norm.sf(abs(z_score)) if pd.notna(z_score) else np.nan,
                "log_verossimilhanca": -model["negative_log_likelihood"],
                "n": model["n"],
                "municipios": model["n_groups"],
                "convergiu": model["success"],
                "gradiente_maximo": model["max_abs_gradient"],
            }
        )
    return pd.DataFrame(rows)


def fit_state_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = []
    comparisons = []
    for year in YEARS:
        frame = panel[panel["ano"].eq(year)].copy()
        y = frame["eleito"].to_numpy(float)
        groups = frame["codigo_municipio"]
        total = frame["log2_despesa_total"].to_numpy(float)
        share = frame["fracao_digital_10pp"].to_numpy(float)
        adoption = frame["adotou_digital"].to_numpy(float)

        model_total = fit_conditional_logit(
            total[:, None],
            y,
            groups,
            groups,
            ["Despesa total: dobro"],
            np.zeros(1),
        )
        coefficients.append(
            coefficient_rows(model_total, "Total", year)
        )

        model_share = fit_conditional_logit(
            np.column_stack([total, share]),
            y,
            groups,
            groups,
            ["Despesa total: dobro", "Fração digital: +10 p.p."],
            np.r_[model_total["beta"], 0.0],
        )
        coefficients.append(
            coefficient_rows(model_share, "Total + fração digital", year)
        )
        likelihood_ratio = 2 * (
            model_total["negative_log_likelihood"]
            - model_share["negative_log_likelihood"]
        )
        comparisons.append(
            {
                "ano": year,
                "comparacao": "Total + fração digital vs. Total",
                "lr": likelihood_ratio,
                "graus_liberdade": 1,
                "p_valor_lr": chi2.sf(likelihood_ratio, 1),
                "n": len(frame),
                "municipios": frame["codigo_municipio"].nunique(),
            }
        )

        if year >= 2020:
            model_full = fit_conditional_logit(
                np.column_stack([total, adoption, share]),
                y,
                groups,
                groups,
                [
                    "Despesa total: dobro",
                    "Qualquer gasto digital",
                    "Fração digital: +10 p.p.",
                ],
                np.r_[model_share["beta"][0], 0.0, model_share["beta"][1]],
            )
            coefficients.append(
                coefficient_rows(
                    model_full,
                    "Total + adoção + fração digital",
                    year,
                )
            )
            likelihood_ratio = 2 * (
                model_total["negative_log_likelihood"]
                - model_full["negative_log_likelihood"]
            )
            comparisons.append(
                {
                    "ano": year,
                    "comparacao": "Total + adoção + fração digital vs. Total",
                    "lr": likelihood_ratio,
                    "graus_liberdade": 2,
                    "p_valor_lr": chi2.sf(likelihood_ratio, 2),
                    "n": len(frame),
                    "municipios": frame["codigo_municipio"].nunique(),
                }
            )
    return pd.concat(coefficients, ignore_index=True), pd.DataFrame(comparisons)


def bootstrap_auc_difference(
    frame: pd.DataFrame,
    score_a: str,
    score_b: str,
    repetitions: int = 5000,
    seed: int = 202407,
) -> dict:
    rng = np.random.default_rng(seed)
    positive = frame.index[frame["eleito"].eq(1)].to_numpy()
    negative = frame.index[frame["eleito"].eq(0)].to_numpy()
    differences = np.empty(repetitions)
    for iteration in range(repetitions):
        sampled = np.r_[
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ]
        sample = frame.loc[sampled]
        differences[iteration] = safe_auc(
            sample["eleito"], sample[score_a]
        ) - safe_auc(sample["eleito"], sample[score_b])
    observed = safe_auc(frame["eleito"], frame[score_a]) - safe_auc(
        frame["eleito"], frame[score_b]
    )
    return {
        "diferenca_auc": observed,
        "ic95_inferior": float(np.quantile(differences, 0.025)),
        "ic95_superior": float(np.quantile(differences, 0.975)),
        "proporcao_bootstrap_maior_zero": float(np.mean(differences > 0)),
        "repeticoes": repetitions,
    }


def pelotas_diagnostics(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    auc_rows = []
    group_rows = []
    tests = []
    cv_rows = []
    score_columns = {
        "Receita": "receita_total_2024",
        "Despesa total": "despesa_total_2024",
        "Despesa não digital": "despesa_nao_digital_2024",
        "Despesa digital": "despesa_digital_2024",
        "Fração digital": "fracao_digital",
    }

    for year in YEARS:
        frame = panel[
            panel["ano"].eq(year) & panel["codigo_municipio"].eq(PELOTAS_CODE)
        ].copy()
        for label, column in score_columns.items():
            auc_rows.append(
                {
                    "ano": year,
                    "indicador": label,
                    "auc": safe_auc(frame["eleito"], frame[column]),
                    "candidatos": len(frame),
                    "eleitos": int(frame["eleito"].sum()),
                }
            )

        bootstrap = bootstrap_auc_difference(
            frame,
            "despesa_total_2024",
            "despesa_nao_digital_2024",
            seed=year,
        )
        tests.append(
            {
                "ano": year,
                "teste": "AUC despesa total − AUC despesa não digital",
                "estimativa": bootstrap["diferenca_auc"],
                "ic95_inferior": bootstrap["ic95_inferior"],
                "ic95_superior": bootstrap["ic95_superior"],
                "p_valor": np.nan,
                "detalhe": (
                    f"{bootstrap['repeticoes']} reamostragens estratificadas; "
                    f"P(Δ>0)={bootstrap['proporcao_bootstrap_maior_zero']:.3f}"
                ),
            }
        )

        contingency = pd.crosstab(frame["eleito"], frame["adotou_digital"]).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0
        )
        odds_ratio, fisher_p = fisher_exact(contingency.to_numpy())
        tests.append(
            {
                "ano": year,
                "teste": "Adoção digital: eleitos vs. não eleitos",
                "estimativa": odds_ratio,
                "ic95_inferior": np.nan,
                "ic95_superior": np.nan,
                "p_valor": fisher_p,
                "detalhe": (
                    f"tabela [[{contingency.iloc[0,0]}, {contingency.iloc[0,1]}], "
                    f"[{contingency.iloc[1,0]}, {contingency.iloc[1,1]}]]"
                ),
            }
        )

        elected_digital = frame.loc[
            frame["eleito"].eq(1), "despesa_digital_2024"
        ]
        non_elected_digital = frame.loc[
            frame["eleito"].eq(0), "despesa_digital_2024"
        ]
        mann = mannwhitneyu(
            elected_digital,
            non_elected_digital,
            alternative="two-sided",
            method="asymptotic",
        )
        tests.append(
            {
                "ano": year,
                "teste": "Despesa digital: Mann–Whitney",
                "estimativa": mann.statistic,
                "ic95_inferior": np.nan,
                "ic95_superior": np.nan,
                "p_valor": mann.pvalue,
                "detalhe": "Teste bicaudal entre eleitos e não eleitos",
            }
        )

        for elected, label in [(0, "Não eleitos"), (1, "Eleitos")]:
            subset = frame[frame["eleito"].eq(elected)]
            group_rows.append(
                {
                    "ano": year,
                    "grupo": label,
                    "candidatos": len(subset),
                    "despesa_total_media_2024": subset[
                        "despesa_total_2024"
                    ].mean(),
                    "despesa_total_mediana_2024": subset[
                        "despesa_total_2024"
                    ].median(),
                    "despesa_digital_media_2024": subset[
                        "despesa_digital_2024"
                    ].mean(),
                    "despesa_digital_mediana_2024": subset[
                        "despesa_digital_2024"
                    ].median(),
                    "fracao_digital_media": subset["fracao_digital"].mean(),
                    "taxa_adocao_digital": subset["adotou_digital"].mean(),
                }
            )

        if year >= 2020:
            features_total = ["log2_despesa_total"]
            features_digital = [
                "log2_despesa_total",
                "adotou_digital",
                "fracao_digital_10pp",
            ]
            splitter = RepeatedStratifiedKFold(
                n_splits=5,
                n_repeats=10,
                random_state=year,
            )
            for repeat_fold, (train, test) in enumerate(
                splitter.split(frame, frame["eleito"]),
                start=1,
            ):
                y_train = frame.iloc[train]["eleito"]
                y_test = frame.iloc[test]["eleito"]
                fold_result = {"ano": year, "repeticao_fold": repeat_fold}
                for label, features in [
                    ("Total", features_total),
                    ("Total + digital", features_digital),
                ]:
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=1.0,
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=year,
                        ),
                    )
                    model.fit(frame.iloc[train][features], y_train)
                    probability = model.predict_proba(frame.iloc[test][features])[:, 1]
                    fold_result[label] = safe_auc(y_test, probability)
                fold_result["diferenca"] = (
                    fold_result["Total + digital"] - fold_result["Total"]
                )
                cv_rows.append(fold_result)

    return (
        pd.DataFrame(auc_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(tests),
        pd.DataFrame(cv_rows),
    )


def municipal_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, code, municipality), frame in panel.groupby(
        ["ano", "codigo_municipio", "municipio"],
        sort=True,
    ):
        elected = frame[frame["eleito"].eq(1)]
        non_elected = frame[frame["eleito"].eq(0)]
        rows.append(
            {
                "ano": year,
                "codigo_municipio": code,
                "municipio": municipality,
                "candidatos": len(frame),
                "eleitos": int(frame["eleito"].sum()),
                "despesa_total_2024": frame["despesa_total_2024"].sum(),
                "despesa_digital_2024": frame["despesa_digital_2024"].sum(),
                "despesa_nao_digital_2024": frame[
                    "despesa_nao_digital_2024"
                ].sum(),
                "fracao_digital_agregada": (
                    frame["despesa_digital_2024"].sum()
                    / frame["despesa_total_2024"].sum()
                    if frame["despesa_total_2024"].sum() > 0
                    else 0.0
                ),
                "taxa_adocao_digital": frame["adotou_digital"].mean(),
                "taxa_adocao_eleitos": elected["adotou_digital"].mean(),
                "taxa_adocao_nao_eleitos": non_elected[
                    "adotou_digital"
                ].mean(),
                "diferenca_adocao_eleitos": (
                    elected["adotou_digital"].mean()
                    - non_elected["adotou_digital"].mean()
                ),
                "fracao_digital_media_eleitos": elected[
                    "fracao_digital"
                ].mean(),
                "fracao_digital_media_nao_eleitos": non_elected[
                    "fracao_digital"
                ].mean(),
                "diferenca_fracao_digital_eleitos": (
                    elected["fracao_digital"].mean()
                    - non_elected["fracao_digital"].mean()
                ),
                "auc_receita": safe_auc(
                    frame["eleito"], frame["receita_total_2024"]
                ),
                "auc_despesa_total": safe_auc(
                    frame["eleito"], frame["despesa_total_2024"]
                ),
                "auc_despesa_nao_digital": safe_auc(
                    frame["eleito"], frame["despesa_nao_digital_2024"]
                ),
                "auc_despesa_digital": safe_auc(
                    frame["eleito"], frame["despesa_digital_2024"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["delta_auc_digital"] = (
        result["auc_despesa_total"] - result["auc_despesa_nao_digital"]
    )
    return result


def add_adjusted_percentiles(indicators: pd.DataFrame) -> pd.DataFrame:
    result = indicators.copy()
    for year in YEARS:
        mask = result["ano"].eq(year)
        year_frame = result.loc[mask].copy()
        log_size = np.log1p(year_frame["candidatos"].to_numpy(float))
        design = np.column_stack([np.ones(len(log_size)), log_size, log_size**2])
        for metric in [
            "fracao_digital_agregada",
            "taxa_adocao_digital",
            "delta_auc_digital",
        ]:
            values = year_frame[metric].to_numpy(float)
            valid = np.isfinite(values)
            fitted = np.full(len(values), np.nan)
            if valid.sum() >= 10:
                beta = np.linalg.lstsq(design[valid], values[valid], rcond=None)[0]
                fitted[valid] = design[valid] @ beta
            residual = values - fitted
            result.loc[mask, f"{metric}_residuo_tamanho"] = residual
            residual_series = pd.Series(residual, index=year_frame.index)
            ranks = residual_series.rank(pct=True, method="average")
            result.loc[mask, f"{metric}_percentil_ajustado"] = ranks
    return result


def state_and_pelotas_summary(
    panel: pd.DataFrame,
    indicators: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        state = panel[panel["ano"].eq(year)]
        pelotas = state[state["codigo_municipio"].eq(PELOTAS_CODE)]
        city_year = indicators[indicators["ano"].eq(year)]
        pelotas_city = city_year[
            city_year["codigo_municipio"].eq(PELOTAS_CODE)
        ].iloc[0]
        for scope, frame in [("Rio Grande do Sul", state), ("Pelotas", pelotas)]:
            total = frame["despesa_total_2024"].sum()
            digital = frame["despesa_digital_2024"].sum()
            row = {
                "ano": year,
                "recorte": scope,
                "candidatos": len(frame),
                "eleitos": int(frame["eleito"].sum()),
                "despesa_total_2024": total,
                "despesa_digital_2024": digital,
                "despesa_impulsionamento_2024": frame[
                    "despesa_impulsionamento_2024"
                ].sum(),
                "despesa_paginas_internet_2024": frame[
                    "despesa_paginas_internet_2024"
                ].sum(),
                "fracao_digital_agregada": digital / total if total else 0.0,
                "taxa_adocao_digital": frame["adotou_digital"].mean(),
                "auc_despesa_total": safe_auc(
                    frame["eleito"], frame["despesa_total_2024"]
                ),
                "auc_despesa_nao_digital": safe_auc(
                    frame["eleito"], frame["despesa_nao_digital_2024"]
                ),
                "auc_despesa_digital": safe_auc(
                    frame["eleito"], frame["despesa_digital_2024"]
                ),
            }
            if scope == "Pelotas":
                row.update(
                    {
                        "percentil_municipal_fracao_digital": percentile_rank(
                            city_year["fracao_digital_agregada"],
                            pelotas_city["fracao_digital_agregada"],
                        ),
                        "percentil_municipal_adocao_digital": percentile_rank(
                            city_year["taxa_adocao_digital"],
                            pelotas_city["taxa_adocao_digital"],
                        ),
                        "percentil_ajustado_fracao_digital": pelotas_city[
                            "fracao_digital_agregada_percentil_ajustado"
                        ],
                        "percentil_ajustado_adocao_digital": pelotas_city[
                            "taxa_adocao_digital_percentil_ajustado"
                        ],
                        "percentil_municipal_delta_auc": percentile_rank(
                            city_year["delta_auc_digital"],
                            pelotas_city["delta_auc_digital"],
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def expense_composition(
    expense_types: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    type_table = expense_types.copy()
    type_table["fator_ipca_2024"] = type_table["ano"].map(
        lambda year: IPCA_TARGET / IPCA_INDEX[year]
    )
    type_table["valor_2024"] = (
        type_table["valor_nominal"] * type_table["fator_ipca_2024"]
    )
    year_totals = type_table.groupby("ano")["valor_2024"].transform("sum")
    type_table["fracao_ano"] = type_table["valor_2024"] / year_totals

    broad = (
        type_table.groupby(["ano", "grupo_despesa"], as_index=False)
        .agg(
            valor_nominal=("valor_nominal", "sum"),
            valor_2024=("valor_2024", "sum"),
            linhas=("linhas", "sum"),
        )
    )
    broad["fracao_ano"] = broad["valor_2024"] / broad.groupby("ano")[
        "valor_2024"
    ].transform("sum")
    broad["grupo_despesa"] = pd.Categorical(
        broad["grupo_despesa"], BROAD_ORDER, ordered=True
    )
    broad = broad.sort_values(["ano", "grupo_despesa"])
    return type_table.sort_values(["ano", "valor_2024"], ascending=[True, False]), broad


def ideology_summary(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel[panel["ideologia"].isin(["Esquerda", "Centro", "Direita"])].copy()
    rows = []
    for (year, ideology, elected), group in frame.groupby(
        ["ano", "ideologia", "eleito"],
        observed=True,
    ):
        total = group["despesa_total_2024"].sum()
        digital = group["despesa_digital_2024"].sum()
        rows.append(
            {
                "ano": year,
                "ideologia": ideology,
                "situacao": "Eleitos" if elected else "Não eleitos",
                "candidatos": len(group),
                "despesa_total_2024": total,
                "despesa_digital_2024": digital,
                "fracao_digital_agregada": digital / total if total else 0.0,
                "taxa_adocao_digital": group["adotou_digital"].mean(),
                "fracao_digital_media": group["fracao_digital"].mean(),
                "despesa_digital_mediana_2024": group[
                    "despesa_digital_2024"
                ].median(),
            }
        )
    return pd.DataFrame(rows)


def supplier_summary(suppliers: pd.DataFrame) -> pd.DataFrame:
    if suppliers.empty:
        return suppliers
    frame = suppliers.copy()
    frame["fator_ipca_2024"] = frame["ano"].map(
        lambda year: IPCA_TARGET / IPCA_INDEX[year]
    )
    frame["valor_2024"] = frame["valor_nominal"] * frame["fator_ipca_2024"]
    result = (
        frame.groupby(["ano", "fornecedor"], as_index=False)
        .agg(
            valor_nominal=("valor_nominal", "sum"),
            valor_2024=("valor_2024", "sum"),
            candidatos=("id_candidato", "nunique"),
            lancamentos=("valor_nominal", "size"),
        )
    )
    result["fracao_digital_pelotas"] = result["valor_2024"] / result.groupby(
        "ano"
    )["valor_2024"].transform("sum")
    return result.sort_values(["ano", "valor_2024"], ascending=[True, False])


def municipal_correlations(
    indicators: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    context_columns = [
        "codigo_municipio",
        "populacao_2022",
        "urbanizacao_2022",
        "pib_per_capita_2023",
        "idhm_2010",
        "gini_renda_2010",
    ]
    context_small = context[context_columns].copy()
    context_small["codigo_municipio"] = normalize_id(
        context_small["codigo_municipio"]
    ).str.zfill(5)
    merged = indicators.merge(context_small, on="codigo_municipio", how="left")
    predictors = [
        "auc_receita",
        "candidatos",
        "populacao_2022",
        "urbanizacao_2022",
        "pib_per_capita_2023",
        "idhm_2010",
        "gini_renda_2010",
    ]
    outcomes = [
        "fracao_digital_agregada",
        "taxa_adocao_digital",
        "delta_auc_digital",
    ]
    rows = []
    for year in YEARS:
        frame = merged[merged["ano"].eq(year)]
        for outcome in outcomes:
            for predictor in predictors:
                valid = frame[[outcome, predictor]].dropna()
                if len(valid) < 10:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(valid[outcome], valid[predictor])
                rows.append(
                    {
                        "ano": year,
                        "indicador_digital": outcome,
                        "variavel_contextual": predictor,
                        "rho_spearman": rho,
                        "p_valor": p_value,
                        "municipios": len(valid),
                    }
                )
    return pd.DataFrame(rows)


def make_figures(
    state_pelotas: pd.DataFrame,
    pelotas_auc: pd.DataFrame,
    pelotas_groups: pd.DataFrame,
    coefficients: pd.DataFrame,
    indicators: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for scope, color, marker in [
        ("Rio Grande do Sul", COLORS["RS"], "o"),
        ("Pelotas", COLORS["Pelotas"], "s"),
    ]:
        frame = state_pelotas[state_pelotas["recorte"].eq(scope)]
        axes[0].plot(
            frame["ano"],
            100 * frame["fracao_digital_agregada"],
            marker=marker,
            linewidth=2.2,
            color=color,
            label=scope,
        )
        axes[1].plot(
            frame["ano"],
            100 * frame["taxa_adocao_digital"],
            marker=marker,
            linewidth=2.2,
            color=color,
            label=scope,
        )
    axes[0].set_title("Parcela digital do gasto total")
    axes[0].set_ylabel("% das despesas")
    axes[1].set_title("Candidatos com algum gasto digital")
    axes[1].set_ylabel("% dos candidatos")
    for axis in axes:
        axis.set_xticks(YEARS)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        axis.legend(frameon=False)
    fig.suptitle(
        "Digitalização das campanhas a vereador: Pelotas e Rio Grande do Sul",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_1_evolucao_digital_pelotas_rs.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    metrics = ["Receita", "Despesa total", "Despesa não digital", "Despesa digital"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for index, year in enumerate(YEARS):
        frame = (
            pelotas_auc[pelotas_auc["ano"].eq(year)]
            .set_index("indicador")
            .reindex(metrics)
        )
        ax.bar(
            x + (index - 1) * width,
            frame["auc"],
            width=width,
            color=COLORS[year],
            label=str(year),
        )
    ax.axhline(0.5, color="#777777", linewidth=1, linestyle="--")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("AUC (0,5 = acaso; 1 = ordenação perfeita)")
    ax.set_title(
        "Pelotas: capacidade de ordenar eleitos acima de não eleitos",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.legend(title="Ano", frameon=False, ncol=3)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_2_auc_pelotas.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for index, year in enumerate(YEARS):
        frame = (
            pelotas_groups[pelotas_groups["ano"].eq(year)]
            .set_index("grupo")
            .reindex(["Não eleitos", "Eleitos"])
        )
        positions = np.arange(2) + (index - 1) * width
        axes[0].bar(
            positions,
            100 * frame["taxa_adocao_digital"],
            width=width,
            color=COLORS[year],
            label=str(year),
        )
        axes[1].bar(
            positions,
            100 * frame["fracao_digital_media"],
            width=width,
            color=COLORS[year],
            label=str(year),
        )
    for axis in axes:
        axis.set_xticks(np.arange(2), ["Não eleitos", "Eleitos"])
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        axis.legend(title="Ano", frameon=False, ncol=3)
    axes[0].set_title("Adoção de gasto digital")
    axes[0].set_ylabel("% do grupo")
    axes[1].set_title("Fração digital média do orçamento")
    axes[1].set_ylabel("% das despesas do candidato")
    fig.suptitle(
        "Pelotas: diferenças entre eleitos e não eleitos",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_3_eleitos_nao_eleitos_pelotas.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    share_forest = coefficients[
        coefficients["modelo"].eq("Total + fração digital")
        & coefficients["parametro"].eq("Fração digital: +10 p.p.")
    ].copy()
    adoption_forest = coefficients[
        coefficients["modelo"].eq("Total + adoção + fração digital")
        & coefficients["parametro"].eq("Qualquer gasto digital")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for ax, forest, xlabel in [
        (
            axes[0],
            adoption_forest,
            "Razão de chances: qualquer gasto digital",
        ),
        (
            axes[1],
            share_forest,
            "Razão de chances por +10 p.p. de gasto digital",
        ),
    ]:
        positions = np.arange(len(forest))
        for position, row in zip(positions, forest.itertuples(index=False)):
            ax.errorbar(
                row.odds_ratio,
                position,
                xerr=np.array(
                    [
                        [row.odds_ratio - row.or_ic95_inferior],
                        [row.or_ic95_superior - row.odds_ratio],
                    ]
                ),
                fmt="o",
                color=COLORS[row.ano],
                ecolor=COLORS[row.ano],
                capsize=4,
                markersize=6,
            )
        ax.axvline(1.0, color="#777777", linestyle="--", linewidth=1)
        ax.set_yticks(positions, forest["ano"].astype(str))
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    axes[0].set_title("Adoção digital")
    axes[1].set_title("Intensidade digital no orçamento")
    fig.suptitle(
        "Associação digital líquida da despesa total no RS",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_4_modelo_digital_rs.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharey=True)
    for axis, year in zip(axes, YEARS):
        frame = indicators[indicators["ano"].eq(year)].copy()
        pelotas = frame[frame["codigo_municipio"].eq(PELOTAS_CODE)].iloc[0]
        axis.scatter(
            100 * frame["fracao_digital_agregada"],
            frame["auc_receita"],
            s=np.clip(frame["candidatos"] / 3, 8, 80),
            color="#B3B3B3",
            alpha=0.55,
            edgecolor="none",
        )
        axis.scatter(
            100 * pelotas["fracao_digital_agregada"],
            pelotas["auc_receita"],
            s=95,
            marker="*",
            color=COLORS["Pelotas"],
            edgecolor="white",
            linewidth=0.8,
            label="Pelotas",
            zorder=5,
        )
        axis.set_title(str(year))
        axis.set_xlabel("Gasto digital (% do total)")
        axis.grid(color="#EAEAEA", linewidth=0.7)
        axis.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("AUC receita–eleição")
    fig.suptitle(
        "Intensidade digital e acoplamento entre receita e eleição nos municípios",
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_5_contexto_municipal.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def write_methodology() -> None:
    methodology = {
        "unidade_analise": "candidatura a vereador com resultado final válido",
        "universo": "todos os 497 municípios do Rio Grande do Sul, 2016, 2020 e 2024",
        "despesa_2020_2024": (
            "despesas contratadas da prestação final; cada linha do arquivo oficial "
            "é preservada, inclusive rateios que repetem SQ_DESPESA entre candidatos"
        ),
        "despesa_2016": (
            "arquivo final de despesas; rubricas iniciadas por 'Baixa de Estimaveis - ' "
            "são incluídas como despesas em espécie estimável e harmonizadas à rubrica-base"
        ),
        "digital_estrito": sorted(DIGITAL_TYPES),
        "nao_incluido_em_digital": (
            "produção de rádio, televisão, vídeo, jingles e slogans; o canal de veiculação "
            "não é identificável apenas pela rubrica e, por isso, forma categoria audiovisual"
        ),
        "deflacionamento": {
            "indice": "IPCA geral no mês da eleição",
            "base": "outubro de 2024",
            "indices": IPCA_INDEX,
        },
        "modelo_principal": (
            "logit condicional exato por município em cada ano, erros-padrão "
            "clusterizados por município; fração digital entra em unidades de 10 pontos "
            "percentuais, controlando log2(1 + despesa total real/1000)"
        ),
        "interpretacao": (
            "associações descritivas e condicionais; os dados observacionais não identificam "
            "efeito causal do gasto digital"
        ),
        "fontes": {
            "2016": "https://dadosabertos.tse.jus.br/dataset/prestacao-de-contas-eleitorais-2016",
            "2020": "https://dadosabertos.tse.jus.br/dataset/prestacao-de-contas-eleitorais-2020",
            "2024": "https://dadosabertos.tse.jus.br/dataset/prestacao-de-contas-eleitorais-2024",
        },
    }
    (OUT / "metodologia.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, dtype={"codigo_municipio": str, "id_candidato": str})
    panel["codigo_municipio"] = normalize_id(panel["codigo_municipio"]).str.zfill(5)
    panel["id_candidato"] = normalize_id(panel["id_candidato"])

    expense_candidates = []
    expense_types = []
    supplier_lines = []
    audit_rows = []
    for year in YEARS:
        candidates, types, suppliers, audit = load_expenses(year)
        expense_candidates.append(candidates)
        expense_types.append(types)
        supplier_lines.append(suppliers)
        audit_rows.append(audit)
    candidate_expenses = pd.concat(expense_candidates, ignore_index=True)
    type_expenses = pd.concat(expense_types, ignore_index=True)
    supplier_lines_frame = pd.concat(supplier_lines, ignore_index=True)

    panel_keys = set(zip(panel["ano"].astype(int), panel["id_candidato"]))
    expense_keys = set(
        zip(candidate_expenses["ano"].astype(int), candidate_expenses["id_candidato"])
    )
    for audit in audit_rows:
        year = audit["ano"]
        panel_year_keys = {key for key in panel_keys if key[0] == year}
        expense_year_keys = {key for key in expense_keys if key[0] == year}
        audit["candidatos_painel_valido"] = len(panel_year_keys)
        audit["candidatos_despesa_fora_painel"] = len(
            expense_year_keys - panel_year_keys
        )
        audit["candidatos_painel_sem_despesa_positiva"] = len(
            panel_year_keys - expense_year_keys
        )
        audit["candidatos_despesa_integrados"] = len(
            panel_year_keys & expense_year_keys
        )

    merge_columns = [
        "ano",
        "id_candidato",
        "despesa_total_nominal",
        "despesa_digital_nominal",
        "despesa_impulsionamento_nominal",
        "despesa_paginas_internet_nominal",
        "despesa_audiovisual_nominal",
        "linhas_despesa",
    ]
    panel = panel.merge(
        candidate_expenses[merge_columns],
        on=["ano", "id_candidato"],
        how="left",
        validate="one_to_one",
    )
    expense_value_columns = [
        "despesa_total_nominal",
        "despesa_digital_nominal",
        "despesa_impulsionamento_nominal",
        "despesa_paginas_internet_nominal",
        "despesa_audiovisual_nominal",
        "linhas_despesa",
    ]
    panel[expense_value_columns] = panel[expense_value_columns].fillna(0.0)
    panel["despesa_nao_digital_nominal"] = (
        panel["despesa_total_nominal"] - panel["despesa_digital_nominal"]
    )
    factor = panel["ano"].map(lambda year: IPCA_TARGET / IPCA_INDEX[year])
    panel["fator_ipca_despesa_2024"] = factor
    for stem in [
        "despesa_total",
        "despesa_digital",
        "despesa_nao_digital",
        "despesa_impulsionamento",
        "despesa_paginas_internet",
        "despesa_audiovisual",
    ]:
        panel[f"{stem}_2024"] = panel[f"{stem}_nominal"] * factor
    panel["fracao_digital"] = np.where(
        panel["despesa_total_nominal"].gt(0),
        panel["despesa_digital_nominal"] / panel["despesa_total_nominal"],
        0.0,
    )
    panel["fracao_digital_10pp"] = panel["fracao_digital"] / 0.10
    panel["adotou_digital"] = panel["despesa_digital_nominal"].gt(0).astype(int)
    panel["log2_despesa_total"] = np.log2(
        1 + panel["despesa_total_2024"] / 1000
    )
    panel["log2_despesa_nao_digital"] = np.log2(
        1 + panel["despesa_nao_digital_2024"] / 1000
    )
    panel["log2_despesa_digital"] = np.log2(
        1 + panel["despesa_digital_2024"] / 1000
    )

    type_table, broad_table = expense_composition(type_expenses)
    indicators = add_adjusted_percentiles(municipal_indicators(panel))
    state_pelotas = state_and_pelotas_summary(panel, indicators)
    coefficients, model_comparisons = fit_state_models(panel)
    pelotas_auc, pelotas_groups, pelotas_tests, pelotas_cv = pelotas_diagnostics(
        panel
    )
    ideology = ideology_summary(panel)
    suppliers = supplier_summary(supplier_lines_frame)
    context = pd.read_csv(CONTEXT_PATH, dtype={"codigo_municipio": str})
    correlations = municipal_correlations(indicators, context)

    audit_table = pd.DataFrame(audit_rows)
    audit_table.to_csv(TABLE_OUT / "auditoria_despesas.csv", index=False)
    type_table.to_csv(TABLE_OUT / "classificacao_tipos_despesa.csv", index=False)
    broad_table.to_csv(TABLE_OUT / "composicao_despesas_estado.csv", index=False)
    state_pelotas.to_csv(TABLE_OUT / "digital_estado_pelotas.csv", index=False)
    coefficients.to_csv(TABLE_OUT / "modelos_digitais_coeficientes.csv", index=False)
    model_comparisons.to_csv(
        TABLE_OUT / "modelos_digitais_comparacoes.csv", index=False
    )
    pelotas_auc.to_csv(TABLE_OUT / "pelotas_diagnostico_auc.csv", index=False)
    pelotas_groups.to_csv(
        TABLE_OUT / "pelotas_eleitos_nao_eleitos.csv", index=False
    )
    pelotas_tests.to_csv(TABLE_OUT / "pelotas_testes.csv", index=False)
    pelotas_cv.to_csv(TABLE_OUT / "pelotas_validacao_cruzada.csv", index=False)
    indicators.to_csv(
        TABLE_OUT / "indicadores_despesa_municipio_ano.csv", index=False
    )
    ideology.to_csv(TABLE_OUT / "perfil_digital_ideologia.csv", index=False)
    suppliers.to_csv(TABLE_OUT / "fornecedores_digitais_pelotas.csv", index=False)
    correlations.to_csv(TABLE_OUT / "correlacoes_digitais.csv", index=False)
    panel.to_csv(DATA_OUT / "painel_candidatos_despesas_digitais.csv.gz", index=False, compression="gzip")

    make_figures(
        state_pelotas,
        pelotas_auc,
        pelotas_groups,
        coefficients,
        indicators,
    )
    write_methodology()

    pelotas_2024 = state_pelotas[
        state_pelotas["ano"].eq(2024)
        & state_pelotas["recorte"].eq("Pelotas")
    ].iloc[0]
    auc_2024 = pelotas_auc[
        pelotas_auc["ano"].eq(2024)
    ].set_index("indicador")["auc"]
    cv_summary = (
        pelotas_cv.groupby("ano")
        .agg(
            auc_cv_total=("Total", "mean"),
            auc_cv_total_digital=("Total + digital", "mean"),
            ganho_auc_cv=("diferenca", "mean"),
            ganho_auc_cv_p10=("diferenca", lambda values: values.quantile(0.10)),
            ganho_auc_cv_p90=("diferenca", lambda values: values.quantile(0.90)),
        )
        .reset_index()
    )
    cv_summary.to_csv(
        TABLE_OUT / "pelotas_validacao_cruzada_resumo.csv", index=False
    )
    summary = {
        "painel_candidatos": len(panel),
        "municipios": panel["codigo_municipio"].nunique(),
        "anos": YEARS,
        "pelotas_2024": {
            "fracao_digital": pelotas_2024["fracao_digital_agregada"],
            "taxa_adocao": pelotas_2024["taxa_adocao_digital"],
            "auc_receita": float(auc_2024["Receita"]),
            "auc_despesa_total": float(auc_2024["Despesa total"]),
            "auc_despesa_nao_digital": float(auc_2024["Despesa não digital"]),
            "auc_despesa_digital": float(auc_2024["Despesa digital"]),
        },
        "conclusao_preliminar": (
            "Pelotas é mais digitalizada que o conjunto estadual em 2020 e 2024. "
            "Em 2024, o gasto digital está fortemente associado à eleição, mas retirar "
            "a parcela digital reduz pouco a AUC da despesa total; portanto, o digital "
            "é um componente relevante, não uma explicação suficiente para o alto "
            "acoplamento entre dinheiro e eleição."
        ),
    }
    (OUT / "resumo_execucao.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
