"""Painel municipal de financiamento eleitoral para vereadores no Rio Grande do Sul.

O script harmoniza os arquivos oficiais do TSE de 2016, 2020 e 2024,
mantém candidaturas com resultado final válido (inclusive receita zero),
classifica as receitas por origem e estima associações entre financiamento
e eleição com interceptos fixos por município-ano.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-painel-rs")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, chi2_contingency, norm, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATA_OUT = OUT / "dados"
TABLE_OUT = OUT / "tabelas"
DATA_OUT.mkdir(parents=True, exist_ok=True)
TABLE_OUT.mkdir(parents=True, exist_ok=True)

YEARS = [2016, 2020, 2024]

# IPCA geral no mês da eleição; dezembro de 1993 = 100.
# Os mesmos fatores foram usados no manuscrito revisado de Pelotas.
IPCA_INDEX = {2016: 4752.86, 2020: 5438.12, 2024: 7036.33}
IPCA_TARGET = IPCA_INDEX[2024]

SOURCE_ORDER = [
    "Pessoas físicas",
    "Recursos próprios",
    "Fundo Partidário",
    "FEFC",
    "Transferências políticas",
    "Financiamento coletivo",
    "Empresas",
    "Outros",
]
PUBLIC_SOURCES = ["Fundo Partidário", "FEFC"]

# Classificação principal de Borges & Vidigal (2023), complementada por
# continuidade partidária e por siglas pequenas não cobertas no apêndice.
LEFT = {
    "PCB",
    "PC do B",
    "PCO",
    "PDT",
    "PPL",
    "PSB",
    "PSOL",
    "PSTU",
    "PT",
    "REDE",
    "UP",
}
CENTER = {
    "AGIR",  # sucessor nominal do PTC
    "AVANTE",
    "CIDADANIA",
    "MDB",
    "PMDB",
    "PHS",
    "PMN",
    "PPS",
    "PROS",
    "PSDB",
    "PTC",
    "PV",
    "SD",
    "SOLIDARIEDADE",
    "PT do B",
}
RIGHT = {
    "DC",
    "DEM",
    "NOVO",
    "PATRIOTA",
    "PL",
    "PMB",
    "PODE",
    "PP",
    "PR",
    "PRB",
    "PRD",  # fusão de PTB e Patriota
    "PRP",
    "PRTB",
    "PSC",
    "PSD",
    "PSDC",
    "PSL",
    "PTB",
    "PTN",
    "REPUBLICANOS",
    "UNIÃO",
}
IDEOLOGY_ORDER = ["Esquerda", "Centro", "Direita"]


def ideology(party: str, sensitivity: bool = False) -> str:
    """Mapeia siglas em três blocos; sensibilidade move centro-esquerda ao centro."""
    if party in LEFT:
        if sensitivity and party in {"PDT", "PSB", "REDE"}:
            return "Centro"
        return "Esquerda"
    if party in CENTER:
        return "Centro"
    if party in RIGHT:
        return "Direita"
    return "Não classificado"


def money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0.0)


def gini(values) -> float:
    x = np.sort(np.maximum(np.asarray(values, dtype=float), 0))
    if len(x) == 0 or np.isclose(x.sum(), 0):
        return np.nan
    n = len(x)
    return 2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n


def safe_share(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else np.nan


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Valores-p ajustados pelo procedimento de Benjamini--Hochberg."""
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return adjusted
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    count = len(ranked)
    raw = ranked * count / np.arange(1, count + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted.loc[order] = np.minimum(monotone, 1.0)
    return adjusted


def classify_source(origin: pd.Series, source: pd.Series) -> pd.Series:
    """Classificação harmonizada, usando primeiro a fonte pública declarada."""
    origin_n = origin.fillna("").str.strip().str.upper()
    source_n = source.fillna("").str.strip().str.upper()
    result = pd.Series("Outros", index=origin.index, dtype="object")

    result.loc[origin_n.eq("RECURSOS DE PESSOAS FÍSICAS")] = "Pessoas físicas"
    result.loc[origin_n.eq("RECURSOS PRÓPRIOS")] = "Recursos próprios"
    result.loc[
        origin_n.isin(
            [
                "RECURSOS DE PARTIDO POLÍTICO",
                "RECURSOS DE OUTROS CANDIDATOS",
            ]
        )
    ] = "Transferências políticas"
    result.loc[origin_n.eq("RECURSOS DE FINANCIAMENTO COLETIVO")] = (
        "Financiamento coletivo"
    )
    result.loc[origin_n.eq("RECURSOS DE PESSOAS JURÍDICAS")] = "Empresas"

    # A fonte declarada prevalece sobre a origem intermediária do repasse.
    result.loc[source_n.str.contains("FUNDO PARTID", regex=False)] = "Fundo Partidário"
    result.loc[source_n.str.contains("FUNDO ESPECIAL", regex=False)] = "FEFC"
    return result


def load_candidates(year: int) -> pd.DataFrame:
    path = ROOT / "tse_downloads" / "candidatos" / f"consulta_cand_{year}_RS.csv"
    frame = pd.read_csv(path, sep=";", encoding="latin1", dtype=str)
    frame = frame[frame["DS_CARGO"].fillna("").str.upper().eq("VEREADOR")].copy()
    frame = frame[
        ~frame["DS_SIT_TOT_TURNO"].fillna("").str.startswith("#NULO")
    ].copy()
    frame["eleito"] = (
        frame["DS_SIT_TOT_TURNO"].fillna("").str.startswith("ELEITO").astype(int)
    )
    frame["ano"] = year
    frame["ideologia"] = frame["SG_PARTIDO"].map(ideology)
    frame["ideologia_sensibilidade"] = frame["SG_PARTIDO"].map(
        lambda party: ideology(party, True)
    )

    keep = [
        "ano",
        "SG_UE",
        "NM_UE",
        "SQ_CANDIDATO",
        "NM_CANDIDATO",
        "NM_URNA_CANDIDATO",
        "SG_PARTIDO",
        "DS_GENERO",
        "DS_COR_RACA",
        "DS_GRAU_INSTRUCAO",
        "DS_OCUPACAO",
        "DS_SIT_TOT_TURNO",
        "eleito",
        "ideologia",
        "ideologia_sensibilidade",
    ]
    out = frame[keep].copy()
    out.columns = [
        "ano",
        "codigo_municipio",
        "municipio",
        "id_candidato",
        "nome_candidato",
        "nome_urna",
        "partido",
        "genero",
        "cor_raca",
        "grau_instrucao",
        "ocupacao",
        "situacao_final",
        "eleito",
        "ideologia",
        "ideologia_sensibilidade",
    ]
    if out["id_candidato"].duplicated().any():
        raise ValueError(f"Identificador de candidato duplicado em {year}.")
    return out


def load_revenues(year: int) -> tuple[pd.DataFrame, dict]:
    if year == 2016:
        path = (
            ROOT
            / "tse_downloads"
            / "extracted"
            / "receitas_candidatos_prestacao_contas_final_2016_RS.txt"
        )
        cols = [
            "Cargo",
            "Sequencial Candidato",
            "Tipo receita",
            "Fonte recurso",
            "Valor receita",
        ]
        cargo_col = "Cargo"
        candidate_col = "Sequencial Candidato"
        origin_col = "Tipo receita"
        source_col = "Fonte recurso"
        value_col = "Valor receita"
    else:
        path = (
            ROOT
            / "tse_downloads"
            / "extracted"
            / f"receitas_candidatos_{year}_RS.csv"
        )
        cols = [
            "DS_CARGO",
            "SQ_CANDIDATO",
            "DS_ORIGEM_RECEITA",
            "DS_FONTE_RECEITA",
            "VR_RECEITA",
        ]
        cargo_col = "DS_CARGO"
        candidate_col = "SQ_CANDIDATO"
        origin_col = "DS_ORIGEM_RECEITA"
        source_col = "DS_FONTE_RECEITA"
        value_col = "VR_RECEITA"

    aggregates = []
    raw_rows = 0
    raw_total = 0.0
    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        dtype=str,
        usecols=cols,
        chunksize=100_000,
    ):
        chunk = chunk[
            chunk[cargo_col].fillna("").str.upper().eq("VEREADOR")
        ].copy()
        if chunk.empty:
            continue
        chunk["valor"] = money(chunk[value_col])
        if (chunk["valor"] < 0).any():
            raise ValueError(f"Receita negativa encontrada em {year}.")
        chunk["fonte"] = classify_source(chunk[origin_col], chunk[source_col])
        chunk["id_candidato"] = chunk[candidate_col]
        raw_rows += len(chunk)
        raw_total += chunk["valor"].sum()
        aggregates.append(
            chunk.groupby(["id_candidato", "fonte"], as_index=False)["valor"].sum()
        )

    long = (
        pd.concat(aggregates, ignore_index=True)
        .groupby(["id_candidato", "fonte"], as_index=False)["valor"]
        .sum()
    )
    wide = long.pivot_table(
        index="id_candidato",
        columns="fonte",
        values="valor",
        fill_value=0.0,
    ).reset_index()
    for source in SOURCE_ORDER:
        if source not in wide:
            wide[source] = 0.0
    audit = {
        "ano": year,
        "arquivo": str(path.relative_to(ROOT)),
        "linhas_vereador_arquivo_receitas": raw_rows,
        "candidatos_com_receita_arquivo": int(wide["id_candidato"].nunique()),
        "total_receitas_vereador_arquivo": float(raw_total),
    }
    return wide[["id_candidato", *SOURCE_ORDER]], audit


def prepare_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.concat(
        [load_candidates(year) for year in YEARS], ignore_index=True
    )
    audit_rows = []
    year_frames = []
    for year in YEARS:
        cand_year = candidates[candidates["ano"].eq(year)].copy()
        revenue, audit = load_revenues(year)
        valid_ids = set(cand_year["id_candidato"])
        revenue_ids = set(revenue["id_candidato"])
        matched = revenue[revenue["id_candidato"].isin(valid_ids)]
        audit.update(
            {
                "candidaturas_validas": len(cand_year),
                "municipios": int(cand_year["codigo_municipio"].nunique()),
                "eleitos": int(cand_year["eleito"].sum()),
                "candidaturas_validas_com_receita": len(valid_ids & revenue_ids),
                "candidaturas_validas_sem_receita": len(valid_ids - revenue_ids),
                "ids_receita_sem_candidatura_valida": len(revenue_ids - valid_ids),
                "total_receitas_candidaturas_validas": float(
                    matched[SOURCE_ORDER].to_numpy().sum()
                ),
            }
        )
        audit_rows.append(audit)
        merged = cand_year.merge(revenue, on="id_candidato", how="left")
        merged[SOURCE_ORDER] = merged[SOURCE_ORDER].fillna(0.0)
        # Valores monetários são observados em centavos. O arredondamento
        # impede que resíduos de soma em ponto flutuante desfaçam empates
        # substantivos no cálculo da AUC municipal.
        merged[SOURCE_ORDER] = merged[SOURCE_ORDER].round(2)
        year_frames.append(merged)

    panel = pd.concat(year_frames, ignore_index=True)
    panel["receita_total_nominal"] = panel[SOURCE_ORDER].sum(axis=1)
    panel["receita_publica_nominal"] = panel[PUBLIC_SOURCES].sum(axis=1)
    panel["fracao_publica_candidato"] = np.where(
        panel["receita_total_nominal"].gt(0),
        panel["receita_publica_nominal"] / panel["receita_total_nominal"],
        np.nan,
    )
    panel["fator_ipca_2024"] = panel["ano"].map(
        lambda year: IPCA_TARGET / IPCA_INDEX[int(year)]
    )
    panel["receita_total_2024"] = (
        panel["receita_total_nominal"] * panel["fator_ipca_2024"]
    )
    panel["log2_receita"] = np.log2(1 + panel["receita_total_2024"] / 1000)
    for source in SOURCE_ORDER:
        panel[f"{source}_2024"] = panel[source] * panel["fator_ipca_2024"]
    return panel, pd.DataFrame(audit_rows)


def city_year_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, code, city), group in panel.groupby(
        ["ano", "codigo_municipio", "municipio"], sort=True
    ):
        values = group["receita_total_2024"].to_numpy(float)
        total_nominal = group["receita_total_nominal"].sum()
        total_real = values.sum()
        n = len(group)
        top_n = max(1, int(np.ceil(0.10 * n)))
        if total_real:
            shares = values / total_real
            positive_shares = shares[shares > 0]
            participation_ratio = 1 / np.sum(shares**2)
            normalized_participation = participation_ratio / n
            entropy = -np.sum(positive_shares * np.log(positive_shares)) / np.log(n)
            top_share = np.sort(values)[-top_n:].sum() / total_real
            elected_share = (
                group.loc[group["eleito"].eq(1), "receita_total_2024"].sum()
                / total_real
            )
        else:
            participation_ratio = np.nan
            normalized_participation = np.nan
            entropy = np.nan
            top_share = np.nan
            elected_share = np.nan
        y = group["eleito"].to_numpy(int)
        auc = roc_auc_score(y, values) if len(np.unique(y)) == 2 else np.nan
        row = {
            "ano": int(year),
            "codigo_municipio": code,
            "municipio": city,
            "candidatos": n,
            "eleitos": int(y.sum()),
            "nao_eleitos": int((1 - y).sum()),
            "receita_zero": int(np.isclose(values, 0).sum()),
            "receita_total_nominal": total_nominal,
            "receita_total_2024": total_real,
            "media_receita_2024": values.mean(),
            "mediana_receita_2024": np.median(values),
            "mediana_eleitos_2024": group.loc[
                group["eleito"].eq(1), "receita_total_2024"
            ].median(),
            "mediana_nao_eleitos_2024": group.loc[
                group["eleito"].eq(0), "receita_total_2024"
            ].median(),
            "gini_receita": gini(values),
            "participacao_top10": top_share,
            "razao_participacao": participation_ratio,
            "razao_participacao_normalizada": normalized_participation,
            "entropia_normalizada": entropy,
            "fracao_receita_eleitos": elected_share,
            "auc_receita_eleicao": auc,
            "fracao_publica": safe_share(
                group["receita_publica_nominal"].sum(), total_nominal
            ),
        }
        for source in SOURCE_ORDER:
            row[f"fracao_{source}"] = safe_share(group[source].sum(), total_nominal)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_fixed_effect_logit(
    X: np.ndarray,
    y: np.ndarray,
    groups,
    clusters,
    parameter_names: list[str],
) -> dict:
    """Logit com interceptos por estrato e covariância clusterizada.

    A estimação conjunta evita construir uma matriz densa de 1.491 dummies.
    A covariância dos coeficientes usa a informação perfilada dos interceptos
    e escores agregados por município.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    group_codes, group_labels = pd.factorize(pd.Series(groups), sort=True)
    cluster_codes, cluster_labels = pd.factorize(pd.Series(clusters), sort=True)
    n_groups = len(group_labels)
    n_params = X.shape[1]
    counts = np.bincount(group_codes, minlength=n_groups)
    successes = np.bincount(group_codes, weights=y, minlength=n_groups)
    rates = np.clip(successes / counts, 1e-5, 1 - 1e-5)
    alpha0 = np.log(rates / (1 - rates))
    initial = np.r_[alpha0, np.zeros(n_params)]

    def objective_gradient(theta):
        alpha = theta[:n_groups]
        beta = theta[n_groups:]
        eta = alpha[group_codes] + X @ beta
        probability = expit(eta)
        objective = np.sum(np.logaddexp(0.0, eta) - y * eta)
        residual = probability - y
        grad_alpha = np.bincount(
            group_codes, weights=residual, minlength=n_groups
        )
        grad_beta = X.T @ residual
        return objective, np.r_[grad_alpha, grad_beta]

    fit = minimize(
        objective_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
    )
    objective, gradient = objective_gradient(fit.x)
    alpha = fit.x[:n_groups]
    beta = fit.x[n_groups:]
    eta = alpha[group_codes] + X @ beta
    probability = expit(eta)
    weight = np.clip(probability * (1 - probability), 1e-12, None)

    h_beta = X.T @ (weight[:, None] * X)
    weight_group = np.bincount(group_codes, weights=weight, minlength=n_groups)
    cross = np.vstack(
        [
            np.bincount(
                group_codes,
                weights=weight * X[:, j],
                minlength=n_groups,
            )
            for j in range(n_params)
        ]
    )
    info_profile = h_beta.copy()
    for group_index in range(n_groups):
        if weight_group[group_index] > 0:
            vector = cross[:, group_index]
            info_profile -= np.outer(vector, vector) / weight_group[group_index]
    bread = np.linalg.pinv(info_profile)

    weighted_group_mean = (cross / weight_group[None, :]).T
    X_tilde = X - weighted_group_mean[group_codes]
    efficient_score = X_tilde * (y - probability)[:, None]
    cluster_score = np.vstack(
        [
            np.bincount(
                cluster_codes,
                weights=efficient_score[:, j],
                minlength=len(cluster_labels),
            )
            for j in range(n_params)
        ]
    ).T
    meat = cluster_score.T @ cluster_score
    correction = len(cluster_labels) / (len(cluster_labels) - 1)
    covariance = correction * bread @ meat @ bread

    max_group_score = np.abs(
        np.bincount(
            group_codes,
            weights=probability - y,
            minlength=n_groups,
        )
    ).max()
    return {
        "success": bool(fit.success),
        "message": str(fit.message),
        "negative_log_likelihood": float(objective),
        "max_abs_gradient": float(np.abs(gradient).max()),
        "max_abs_group_score": float(max_group_score),
        "beta": beta,
        "covariance": covariance,
        "standard_error": np.sqrt(np.maximum(np.diag(covariance), 0)),
        "alpha": alpha,
        "probability": probability,
        "group_codes": group_codes,
        "group_labels": group_labels,
        "parameter_names": parameter_names,
        "n": len(y),
        "n_groups": n_groups,
        "n_clusters": len(cluster_labels),
    }


def fit_conditional_logit(
    X: np.ndarray,
    y: np.ndarray,
    groups,
    clusters,
    parameter_names: list[str],
    initial: np.ndarray,
) -> dict:
    """Logit condicional exato por estrato, sem viés dos interceptos fixos.

    O denominador da verossimilhança condicional é calculado por programação
    dinâmica como um polinômio simétrico elementar. Os escores são agregados
    por município para a covariância sanduíche clusterizada.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    group_codes, group_labels = pd.factorize(pd.Series(groups), sort=True)
    cluster_codes, cluster_labels = pd.factorize(pd.Series(clusters), sort=True)
    order = np.argsort(group_codes, kind="stable")
    breaks = np.r_[
        0,
        np.flatnonzero(np.diff(group_codes[order])) + 1,
        len(group_codes),
    ]
    group_indices = [
        order[breaks[index] : breaks[index + 1]]
        for index in range(len(breaks) - 1)
    ]

    def group_terms(X_group, y_group, beta, need_hessian=False):
        selected = int(y_group.sum())
        parameter_count = len(beta)
        denominator = np.zeros(selected + 1)
        denominator[0] = 1.0
        derivative = np.zeros((selected + 1, parameter_count))
        if need_hessian:
            second = np.zeros(
                (selected + 1, parameter_count, parameter_count)
            )
        log_scale = 0.0

        for x_row in X_group:
            weight = math.exp(float(x_row @ beta))
            old_denominator = denominator.copy()
            old_derivative = derivative.copy()
            denominator[1:] = (
                old_denominator[1:] + weight * old_denominator[:-1]
            )
            derivative[1:] = old_derivative[1:] + weight * (
                old_derivative[:-1]
                + old_denominator[:-1, None] * x_row
            )
            if need_hessian:
                old_second = second.copy()
                outer_x = np.outer(x_row, x_row)
                second[1:] = old_second[1:] + weight * (
                    old_second[:-1]
                    + old_derivative[:-1, :, None] * x_row[None, None, :]
                    + x_row[None, :, None] * old_derivative[:-1, None, :]
                    + old_denominator[:-1, None, None] * outer_x
                )
            scale = denominator.max()
            if scale > 1e100 or scale < 1e-100:
                denominator /= scale
                derivative /= scale
                if need_hessian:
                    second /= scale
                log_scale += math.log(scale)

        final_denominator = denominator[selected]
        expected_sum = derivative[selected] / final_denominator
        observed_sum = X_group[y_group == 1].sum(axis=0)
        negative_log_likelihood = (
            math.log(final_denominator)
            + log_scale
            - float(y_group @ (X_group @ beta))
        )
        gradient = expected_sum - observed_sum
        if not need_hessian:
            return negative_log_likelihood, gradient
        hessian = (
            second[selected] / final_denominator
            - np.outer(expected_sum, expected_sum)
        )
        return negative_log_likelihood, gradient, hessian

    def objective_gradient(beta):
        objective = 0.0
        gradient = np.zeros_like(beta)
        for indices in group_indices:
            value, group_gradient = group_terms(
                X[indices], y[indices], beta, need_hessian=False
            )
            objective += value
            gradient += group_gradient
        return objective, gradient

    fit = minimize(
        objective_gradient,
        np.asarray(initial, dtype=float),
        jac=True,
        method="BFGS",
        options={"maxiter": 100, "gtol": 1e-4},
    )
    objective, gradient = objective_gradient(fit.x)

    information = np.zeros((X.shape[1], X.shape[1]))
    group_scores = []
    for indices in group_indices:
        _, group_gradient, group_hessian = group_terms(
            X[indices], y[indices], fit.x, need_hessian=True
        )
        information += group_hessian
        group_scores.append(-group_gradient)
    group_scores = np.asarray(group_scores)

    # Cada estrato é município-ano; a ordem dos rótulos coincide com os
    # grupos fatorizados. Três escores anuais são somados em cada município.
    group_cluster_codes = np.array(
        [
            cluster_codes[indices[0]]
            for indices in group_indices
        ],
        dtype=int,
    )
    cluster_score = np.vstack(
        [
            np.bincount(
                group_cluster_codes,
                weights=group_scores[:, parameter],
                minlength=len(cluster_labels),
            )
            for parameter in range(X.shape[1])
        ]
    ).T
    bread = np.linalg.pinv(information)
    meat = cluster_score.T @ cluster_score
    correction = len(cluster_labels) / (len(cluster_labels) - 1)
    covariance = correction * bread @ meat @ bread
    return {
        "success": bool(
            fit.success or np.abs(gradient).max() < 1e-3
        ),
        "message": str(fit.message),
        "negative_log_likelihood": float(objective),
        "max_abs_gradient": float(np.abs(gradient).max()),
        "beta": fit.x,
        "covariance": covariance,
        "standard_error": np.sqrt(np.maximum(np.diag(covariance), 0)),
        "parameter_names": parameter_names,
        "n": len(y),
        "n_groups": len(group_labels),
        "n_clusters": len(cluster_labels),
    }


def coefficient_table(model: dict, model_name: str) -> pd.DataFrame:
    rows = []
    for name, beta, se in zip(
        model["parameter_names"], model["beta"], model["standard_error"]
    ):
        z = beta / se
        rows.append(
            {
                "modelo": model_name,
                "parametro": name,
                "beta": beta,
                "erro_padrao_cluster": se,
                "odds_ratio": math.exp(beta),
                "or_ic95_inferior": math.exp(beta - 1.96 * se),
                "or_ic95_superior": math.exp(beta + 1.96 * se),
                "p_valor": 2 * norm.sf(abs(z)),
                "n": model["n"],
                "estratos_municipio_ano": model["n_groups"],
                "clusters_municipio": model["n_clusters"],
                "convergiu": model["success"],
                "gradiente_maximo": model["max_abs_gradient"],
            }
        )
    return pd.DataFrame(rows)


def estimate_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = panel["eleito"].to_numpy(float)
    city_year = panel["ano"].astype(str) + "_" + panel["codigo_municipio"]
    clusters = panel["codigo_municipio"]
    revenue = panel["log2_receita"].to_numpy(float)

    X_year = np.column_stack(
        [revenue * panel["ano"].eq(year).to_numpy(float) for year in YEARS]
    )
    fe_model_year = fit_fixed_effect_logit(
        X_year,
        y,
        city_year,
        clusters,
        [f"receita_{year}" for year in YEARS],
    )
    fe_common_model = fit_fixed_effect_logit(
        revenue[:, None],
        y,
        city_year,
        clusters,
        ["receita_comum"],
    )
    model_year = fit_conditional_logit(
        X_year,
        y,
        city_year,
        clusters,
        [f"receita_{year}" for year in YEARS],
        initial=fe_model_year["beta"],
    )
    common_model = fit_conditional_logit(
        revenue[:, None],
        y,
        city_year,
        clusters,
        ["receita_comum"],
        initial=fe_common_model["beta"],
    )
    global_lr = 2 * (
        common_model["negative_log_likelihood"]
        - model_year["negative_log_likelihood"]
    )
    global_p = chi2.sf(global_lr, len(YEARS) - 1)

    coefficients = [
        coefficient_table(
            model_year,
            "Logit condicional — receita total por ano",
        ),
        coefficient_table(
            common_model,
            "Logit condicional — inclinação comum",
        ),
    ]
    contrast_rows = [
        {
            "contraste": "Teste global: inclinações iguais",
            "estimativa": global_lr,
            "erro_padrao": np.nan,
            "estatistica": global_lr,
            "graus_liberdade": 2,
            "p_valor": global_p,
            "tipo_teste": "Razão de verossimilhanças",
        }
    ]
    for year_a, year_b in [(2020, 2016), (2024, 2016), (2024, 2020)]:
        vector = np.zeros(3)
        vector[YEARS.index(year_a)] = 1
        vector[YEARS.index(year_b)] = -1
        estimate = vector @ model_year["beta"]
        se = math.sqrt(vector @ model_year["covariance"] @ vector)
        contrast_rows.append(
            {
                "contraste": f"{year_a} menos {year_b}",
                "estimativa": estimate,
                "erro_padrao": se,
                "estatistica": estimate / se,
                "graus_liberdade": 1,
                "p_valor": 2 * norm.sf(abs(estimate / se)),
                "tipo_teste": "Wald, EP clusterizado por município",
            }
        )

    # Modelos multivariáveis por origem; uma coluna fonte-ano é incluída
    # somente quando a fonte possui valor positivo no respectivo pleito.
    source_columns = []
    source_names = []
    for source in [
        "Pessoas físicas",
        "Recursos próprios",
        "Fundo Partidário",
        "FEFC",
        "Transferências políticas",
    ]:
        source_real = panel[f"{source}_2024"].to_numpy(float)
        source_log = np.log2(1 + source_real / 1000)
        for year in YEARS:
            mask = panel["ano"].eq(year).to_numpy(float)
            column = source_log * mask
            if np.any(column > 0):
                source_columns.append(column)
                source_names.append(f"{source} — {year}")
    source_model = fit_fixed_effect_logit(
        np.column_stack(source_columns),
        y,
        city_year,
        clusters,
        source_names,
    )
    coefficients.append(
        coefficient_table(
            source_model,
            "Logit FE exploratório — origens simultâneas",
        )
    )

    # Curvas padronizadas: média municipal não ponderada das probabilidades
    # previstas pelos interceptos fixos em uma mesma grade de receita real.
    max_grid = panel["receita_total_2024"].quantile(0.995)
    grid = np.r_[0.0, np.geomspace(50, max(max_grid, 1000), 160)]
    curve_rows = []
    label_to_alpha = {
        str(label): alpha
        for label, alpha in zip(
            fe_model_year["group_labels"], fe_model_year["alpha"]
        )
    }
    for year_index, year in enumerate(YEARS):
        alpha_year = np.array(
            [
                label_to_alpha[f"{year}_{code}"]
                for code in sorted(
                    panel.loc[panel["ano"].eq(year), "codigo_municipio"].unique()
                )
            ]
        )
        beta = fe_model_year["beta"][year_index]
        for value in grid:
            transformed = np.log2(1 + value / 1000)
            probabilities = expit(alpha_year + beta * transformed)
            curve_rows.append(
                {
                    "ano": year,
                    "receita_2024": value,
                    "probabilidade_media_municipal": probabilities.mean(),
                    "probabilidade_mediana_municipal": np.median(probabilities),
                }
            )
    curves = pd.DataFrame(curve_rows)
    validation_rows = []
    for year_index, year in enumerate(YEARS):
        validation_rows.append(
            {
                "ano": year,
                "beta_logit_condicional": model_year["beta"][year_index],
                "beta_logit_dummies": fe_model_year["beta"][year_index],
                "diferenca_absoluta": abs(
                    model_year["beta"][year_index]
                    - fe_model_year["beta"][year_index]
                ),
            }
        )
    return (
        pd.concat(coefficients, ignore_index=True),
        pd.DataFrame(contrast_rows),
        curves,
        pd.DataFrame(validation_rows),
    )


def city_level_tests(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for year, group in metrics.groupby("ano"):
        eligible = group[
            group["candidatos"].ge(20)
            & group["eleitos"].ge(5)
            & group["nao_eleitos"].ge(5)
        ].copy()
        for x, y, label in [
            (
                "fracao_publica",
                "auc_receita_eleicao",
                "Fração pública × AUC receita-eleição",
            ),
            ("fracao_publica", "gini_receita", "Fração pública × Gini"),
            (
                "candidatos",
                "auc_receita_eleicao",
                "Número de candidatos × AUC",
            ),
        ]:
            valid = eligible[[x, y]].dropna()
            rho, p_value = spearmanr(valid[x], valid[y])
            rows.append(
                {
                    "ano": int(year),
                    "relacao": label,
                    "n_municipios": len(valid),
                    "rho_spearman": rho,
                    "p_valor": p_value,
                }
            )

    wide = metrics.pivot(
        index=["codigo_municipio", "municipio"],
        columns="ano",
        values=[
            "fracao_publica",
            "gini_receita",
            "auc_receita_eleicao",
            "entropia_normalizada",
        ],
    )
    changes = []
    for metric in [
        "fracao_publica",
        "gini_receita",
        "auc_receita_eleicao",
        "entropia_normalizada",
    ]:
        before = wide[(metric, 2016)]
        after = wide[(metric, 2024)]
        valid = pd.concat([before, after], axis=1).dropna()
        delta = valid.iloc[:, 1] - valid.iloc[:, 0]
        test = wilcoxon(
            valid.iloc[:, 1],
            valid.iloc[:, 0],
            alternative="two-sided",
            zero_method="wilcox",
        )
        changes.append(
            {
                "indicador": metric,
                "n_municipios_pareados": len(valid),
                "mediana_2016": valid.iloc[:, 0].median(),
                "mediana_2024": valid.iloc[:, 1].median(),
                "mediana_variacao_2024_menos_2016": delta.median(),
                "percentual_municipios_com_aumento": 100 * delta.gt(0).mean(),
                "estatistica_wilcoxon": test.statistic,
                "p_valor": test.pvalue,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(changes)


def ideology_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (year, ideol), group in panel.groupby(["ano", "ideologia"], sort=True):
        total = group["receita_total_nominal"].sum()
        for source in SOURCE_ORDER:
            values = group[source]
            rows.append(
                {
                    "ano": int(year),
                    "ideologia": ideol,
                    "fonte": source,
                    "candidatos": len(group),
                    "eleitos": int(group["eleito"].sum()),
                    "taxa_eleicao": group["eleito"].mean(),
                    "recebedores": int(values.gt(0).sum()),
                    "taxa_acesso": values.gt(0).mean(),
                    "valor_total_nominal": values.sum(),
                    "valor_eleitos_nominal": values[group["eleito"].eq(1)].sum(),
                    "participacao_fonte_para_eleitos": safe_share(
                        values[group["eleito"].eq(1)].sum(), values.sum()
                    ),
                    "participacao_na_receita_bloco": safe_share(
                        values.sum(), total
                    ),
                    "media_por_candidato": values.mean(),
                    "mediana_entre_recebedores": (
                        values[values.gt(0)].median()
                        if values.gt(0).any()
                        else np.nan
                    ),
                }
            )
    profile = pd.DataFrame(rows)

    tests = []
    for year in YEARS:
        year_data = panel[panel["ano"].eq(year)]
        for source in SOURCE_ORDER:
            table = []
            for ideol in IDEOLOGY_ORDER:
                values = year_data.loc[year_data["ideologia"].eq(ideol), source]
                table.append([int(values.gt(0).sum()), int(values.eq(0).sum())])
            table = np.asarray(table)
            if np.any(table.sum(axis=1).eq(0)) if isinstance(table, pd.DataFrame) else False:
                continue
            if table[:, 0].sum() == 0:
                tests.append(
                    {
                        "ano": year,
                        "fonte": source,
                        "chi2": np.nan,
                        "p_valor": np.nan,
                        "v_cramer": np.nan,
                        "observacao": "Nenhum recebedor",
                    }
                )
                continue
            statistic, p_value, _, _ = chi2_contingency(table)
            n = table.sum()
            v_cramer = math.sqrt(statistic / (n * min(table.shape[0] - 1, 1)))
            tests.append(
                {
                    "ano": year,
                    "fonte": source,
                    "chi2": statistic,
                    "p_valor": p_value,
                    "v_cramer": v_cramer,
                    "observacao": "",
                }
            )
    tests = pd.DataFrame(tests)
    tests["p_bh"] = benjamini_hochberg(tests["p_valor"])
    return profile, tests


def ideology_sensitivity_profile(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    schemes = {
        "Principal": "ideologia",
        "PDT, PSB e REDE no centro": "ideologia_sensibilidade",
    }
    for scheme, ideology_column in schemes.items():
        for (year, ideol), group in panel.groupby(
            ["ano", ideology_column], sort=True
        ):
            total = group["receita_total_nominal"].sum()
            for source in SOURCE_ORDER:
                rows.append(
                    {
                        "classificacao": scheme,
                        "ano": int(year),
                        "ideologia": ideol,
                        "fonte": source,
                        "candidatos": len(group),
                        "valor_total_nominal": group[source].sum(),
                        "participacao_na_receita_bloco": safe_share(
                            group[source].sum(), total
                        ),
                        "taxa_acesso": group[source].gt(0).mean(),
                    }
                )
    return pd.DataFrame(rows)


def source_composition(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in panel.groupby("ano"):
        total = group["receita_total_nominal"].sum()
        for source in SOURCE_ORDER:
            values = group[source]
            rows.append(
                {
                    "ano": int(year),
                    "fonte": source,
                    "valor_total_nominal": values.sum(),
                    "participacao_estadual": safe_share(values.sum(), total),
                    "candidatos_recebedores": int(values.gt(0).sum()),
                    "taxa_acesso": values.gt(0).mean(),
                    "mediana_entre_recebedores": (
                        values[values.gt(0)].median()
                        if values.gt(0).any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def competition_strata(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = ["Q1: menos candidatos", "Q2", "Q3", "Q4: mais candidatos"]
    for year, group in metrics.groupby("ano"):
        group = group.copy()
        group["quartil_competicao"] = pd.qcut(
            group["candidatos"],
            q=4,
            labels=labels,
            duplicates="drop",
        )
        for quartile, stratum in group.groupby(
            "quartil_competicao", observed=True
        ):
            rows.append(
                {
                    "ano": int(year),
                    "quartil_competicao": str(quartile),
                    "municipios": len(stratum),
                    "min_candidatos": int(stratum["candidatos"].min()),
                    "max_candidatos": int(stratum["candidatos"].max()),
                    "mediana_candidatos": stratum["candidatos"].median(),
                    "mediana_fracao_publica": stratum["fracao_publica"].median(),
                    "mediana_gini": stratum["gini_receita"].median(),
                    "mediana_auc": stratum["auc_receita_eleicao"].median(),
                }
            )
    return pd.DataFrame(rows)


def municipality_changes(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["codigo_municipio", "municipio"]
    wide = metrics.pivot(
        index=keys,
        columns="ano",
        values=[
            "candidatos",
            "fracao_publica",
            "gini_receita",
            "auc_receita_eleicao",
            "entropia_normalizada",
        ],
    )
    wide.columns = [
        f"{metric}_{year}" for metric, year in wide.columns.to_flat_index()
    ]
    wide = wide.reset_index()
    for metric in [
        "fracao_publica",
        "gini_receita",
        "auc_receita_eleicao",
        "entropia_normalizada",
    ]:
        wide[f"variacao_{metric}_2024_menos_2016"] = (
            wide[f"{metric}_2024"] - wide[f"{metric}_2016"]
        )
    return wide


def pelotas_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, year_data in metrics.groupby("ano"):
        pelotas = year_data[year_data["municipio"].eq("PELOTAS")]
        if len(pelotas) != 1:
            raise ValueError(f"Pelotas não identificado unicamente em {year}.")
        pelotas = pelotas.iloc[0]
        for indicator in [
            "candidatos",
            "fracao_publica",
            "gini_receita",
            "entropia_normalizada",
            "auc_receita_eleicao",
        ]:
            state_values = year_data[indicator].dropna()
            value = pelotas[indicator]
            percentile = 100 * (
                (state_values < value).sum()
                + 0.5 * (state_values == value).sum()
            ) / len(state_values)
            rows.append(
                {
                    "ano": int(year),
                    "indicador": indicator,
                    "valor_pelotas": value,
                    "mediana_rs": state_values.median(),
                    "percentil_pelotas_no_rs": percentile,
                }
            )
    return pd.DataFrame(rows)


def state_summary(panel: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in panel.groupby("ano"):
        city = metrics[metrics["ano"].eq(year)]
        total = group["receita_total_nominal"].sum()
        values = group["receita_total_2024"].to_numpy()
        rows.append(
            {
                "ano": int(year),
                "municipios": int(group["codigo_municipio"].nunique()),
                "candidatos": len(group),
                "eleitos": int(group["eleito"].sum()),
                "taxa_eleicao": group["eleito"].mean(),
                "receita_zero": int(group["receita_total_nominal"].eq(0).sum()),
                "receita_total_nominal": total,
                "receita_total_2024": values.sum(),
                "mediana_candidato_2024": np.median(values),
                "gini_estadual": gini(values),
                "fracao_publica_estadual": safe_share(
                    group["receita_publica_nominal"].sum(), total
                ),
                "mediana_fracao_publica_municipal": city["fracao_publica"].median(),
                "mediana_gini_municipal": city["gini_receita"].median(),
                "mediana_auc_municipal": city["auc_receita_eleicao"].median(),
                "media_auc_municipal": city["auc_receita_eleicao"].mean(),
                "auc_estadual_sem_estratificacao": roc_auc_score(
                    group["eleito"], values
                ),
            }
        )
    return pd.DataFrame(rows)


def write_methodology() -> None:
    methodology = {
        "escopo": (
            "Candidaturas a vereador nos 497 municípios do Rio Grande do Sul, "
            "eleições municipais de 2016, 2020 e 2024."
        ),
        "unidade_analise": "candidato; indicadores agregados por município-ano",
        "inclusao": (
            "DS_CARGO = VEREADOR e DS_SIT_TOT_TURNO diferente de #NULO; "
            "candidaturas sem receita são mantidas com valor zero."
        ),
        "resultado": "eleito = DS_SIT_TOT_TURNO começa por ELEITO",
        "precisao_monetaria": (
            "totais candidato-fonte arredondados a centavos antes dos "
            "indicadores, preservando empates substantivos na AUC"
        ),
        "deflacao": {
            "base": "reais de outubro de 2024",
            "indice": "IPCA geral, dezembro de 1993 = 100",
            "valores": IPCA_INDEX,
        },
        "fontes": {
            "Pessoas físicas": "DS/Tipo de origem = Recursos de pessoas físicas",
            "Recursos próprios": "DS/Tipo de origem = Recursos próprios",
            "Fundo Partidário": "Fonte declarada contém Fundo Partidário",
            "FEFC": "Fonte declarada contém Fundo Especial",
            "Transferências políticas": (
                "origem partido político ou outros candidatos, quando a fonte "
                "não é Fundo Partidário nem FEFC"
            ),
            "Financiamento coletivo": (
                "origem Recursos de Financiamento Coletivo"
            ),
            "Empresas": "origem Recursos de pessoas jurídicas",
            "Outros": "demais origens",
        },
        "observacao_empresas": (
            "Não há registros classificados pelo TSE como recursos de pessoas "
            "jurídicas para vereador no RS em 2016; CNPJs de diretórios "
            "partidários e de outros candidatos não são tratados como empresas."
        ),
        "modelo_principal": (
            "regressão logística condicional exata por município-ano; "
            "receita transformada como log2(1 + receita_real/1000); erros-padrão "
            "clusterizados por município."
        ),
        "modelo_origens": (
            "regressão logística exploratória com interceptos fixos por "
            "município-ano e origens simultâneas; erros-padrão clusterizados "
            "por município."
        ),
        "curvas_probabilidade": (
            "padronização descritiva obtida do logit com dummies de "
            "município-ano, usando a média não ponderada dos 497 municípios."
        ),
        "auc_municipal": (
            "probabilidade de um eleito escolhido ao acaso ter receita maior "
            "que um não eleito do mesmo município-ano; empates recebem 0,5."
        ),
        "ideologia": (
            "Borges e Vidigal (2023); siglas pequenas e sucessoras são "
            "documentadas na tabela de classificação. A sensibilidade move "
            "PDT, PSB e REDE da esquerda para o centro."
        ),
        "arquivos_tse": {
            str(year): {
                "candidaturas": (
                    f"tse_downloads/candidatos/consulta_cand_{year}_RS.csv"
                ),
                "receitas": (
                    "tse_downloads/extracted/"
                    + (
                        "receitas_candidatos_prestacao_contas_final_2016_RS.txt"
                        if year == 2016
                        else f"receitas_candidatos_{year}_RS.csv"
                    )
                ),
            }
            for year in YEARS
        },
    }
    (OUT / "metodologia.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    raw_candidate_paths = [
        ROOT
        / "tse_downloads"
        / "candidatos"
        / f"consulta_cand_{year}_RS.csv"
        for year in YEARS
    ]
    raw_revenue_paths = [
        (
            ROOT
            / "tse_downloads"
            / "extracted"
            / "receitas_candidatos_prestacao_contas_final_2016_RS.txt"
            if year == 2016
            else ROOT
            / "tse_downloads"
            / "extracted"
            / f"receitas_candidatos_{year}_RS.csv"
        )
        for year in YEARS
    ]
    derived_panel_path = DATA_OUT / "painel_candidatos_rs.csv.gz"
    audit_path = TABLE_OUT / "auditoria_importacao.csv"
    if all(path.exists() for path in [*raw_candidate_paths, *raw_revenue_paths]):
        panel, audit = prepare_panel()
    elif derived_panel_path.exists() and audit_path.exists():
        panel = pd.read_csv(
            derived_panel_path,
            encoding="utf-8-sig",
            low_memory=False,
            dtype={
                "id_candidato": "string",
                "codigo_municipio": "string",
            },
        )
        panel[SOURCE_ORDER] = panel[SOURCE_ORDER].round(2)
        panel["receita_total_nominal"] = (
            panel[SOURCE_ORDER].sum(axis=1).round(2)
        )
        panel["receita_publica_nominal"] = (
            panel[PUBLIC_SOURCES].sum(axis=1).round(2)
        )
        panel["fracao_publica_candidato"] = np.where(
            panel["receita_total_nominal"].gt(0),
            panel["receita_publica_nominal"]
            / panel["receita_total_nominal"],
            np.nan,
        )
        # Reconstruct transformed monetary columns from the two-decimal
        # nominal amounts. This preserves exact ties used by municipal AUC
        # calculations after a CSV round trip.
        panel["fator_ipca_2024"] = panel["ano"].map(
            lambda year: IPCA_TARGET / IPCA_INDEX[int(year)]
        )
        panel["receita_total_2024"] = (
            panel["receita_total_nominal"] * panel["fator_ipca_2024"]
        )
        panel["log2_receita"] = np.log2(
            1 + panel["receita_total_2024"] / 1000
        )
        for source in SOURCE_ORDER:
            panel[f"{source}_2024"] = (
                panel[source] * panel["fator_ipca_2024"]
            )
        audit = pd.read_csv(audit_path)
        print(
            "Arquivos TSE brutos não encontrados; recalculando os resultados "
            "a partir do painel candidato a candidato fornecido."
        )
    else:
        missing = [
            str(path)
            for path in [*raw_candidate_paths, *raw_revenue_paths]
            if not path.exists()
        ]
        raise FileNotFoundError(
            "Não foram encontrados todos os arquivos TSE brutos nem o painel "
            f"derivado em {derived_panel_path}. Ausentes: {missing}"
        )
    if set(panel["ideologia"]) != set(IDEOLOGY_ORDER):
        unknown = sorted(
            panel.loc[
                panel["ideologia"].eq("Não classificado"), "partido"
            ].unique()
        )
        if unknown:
            raise ValueError(f"Partidos sem classificação: {unknown}")

    metrics = city_year_metrics(panel)
    coefficients, contrasts, curves, model_validation = estimate_models(panel)
    correlations, paired_changes = city_level_tests(metrics)
    ideology_profile, ideology_tests = ideology_tables(panel)
    ideology_sensitivity = ideology_sensitivity_profile(panel)
    composition = source_composition(panel)
    strata = competition_strata(metrics)
    changes_by_city = municipality_changes(metrics)
    pelotas_vs_state = pelotas_comparison(metrics)
    summary = state_summary(panel, metrics)

    party_map = (
        panel[
            [
                "ano",
                "partido",
                "ideologia",
                "ideologia_sensibilidade",
            ]
        ]
        .drop_duplicates()
        .sort_values(["ano", "ideologia", "partido"])
    )
    party_map["criterio"] = np.where(
        party_map["partido"].isin(
            {
                "AGIR",
                "PCO",
                "PPL",
                "PMB",
                "PRD",
                "UP",
            }
        ),
        "Complemento por continuidade partidária/literatura; testar exclusão",
        "Borges & Vidigal (2023), Tabela A.1; equivalência histórica",
    )

    # Auditorias numéricas antes da exportação.
    if not np.allclose(
        panel["receita_total_nominal"].to_numpy(),
        panel[SOURCE_ORDER].sum(axis=1).to_numpy(),
    ):
        raise AssertionError("As fontes não recompõem a receita total.")
    if len(metrics) != 497 * 3:
        raise AssertionError(f"Esperados 1.491 municípios-ano; obtidos {len(metrics)}.")
    if panel["id_candidato"].duplicated().any():
        raise AssertionError("IDs de candidato duplicados no painel.")

    panel.to_csv(
        DATA_OUT / "painel_candidatos_rs.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    audit.to_csv(TABLE_OUT / "auditoria_importacao.csv", index=False)
    metrics.to_csv(TABLE_OUT / "indicadores_municipio_ano.csv", index=False)
    coefficients.to_csv(TABLE_OUT / "modelos_efeitos_fixos.csv", index=False)
    contrasts.to_csv(TABLE_OUT / "contrastes_temporais.csv", index=False)
    curves.to_csv(TABLE_OUT / "curvas_probabilidade.csv", index=False)
    model_validation.to_csv(TABLE_OUT / "validacao_modelos.csv", index=False)
    correlations.to_csv(TABLE_OUT / "correlacoes_municipais.csv", index=False)
    paired_changes.to_csv(TABLE_OUT / "mudancas_pareadas.csv", index=False)
    changes_by_city.to_csv(TABLE_OUT / "mudancas_por_municipio.csv", index=False)
    ideology_profile.to_csv(TABLE_OUT / "perfil_ideologia_fonte.csv", index=False)
    ideology_sensitivity.to_csv(
        TABLE_OUT / "sensibilidade_ideologia.csv", index=False
    )
    ideology_tests.to_csv(TABLE_OUT / "testes_ideologia_acesso.csv", index=False)
    composition.to_csv(TABLE_OUT / "composicao_fontes_estado.csv", index=False)
    strata.to_csv(TABLE_OUT / "estratos_competicao.csv", index=False)
    pelotas_vs_state.to_csv(TABLE_OUT / "pelotas_vs_rs.csv", index=False)
    summary.to_csv(TABLE_OUT / "resumo_estadual.csv", index=False)
    party_map.to_csv(TABLE_OUT / "classificacao_partidaria.csv", index=False)
    write_methodology()

    run_summary = {
        "linhas_candidatos": len(panel),
        "municipios_ano": len(metrics),
        "anos": YEARS,
        "municipios_por_ano": {
            str(year): int(
                panel.loc[panel["ano"].eq(year), "codigo_municipio"].nunique()
            )
            for year in YEARS
        },
        "modelos_convergiram": bool(coefficients["convergiu"].all()),
        "arquivos_gerados": sorted(
            str(path.relative_to(OUT))
            for path in OUT.rglob("*")
            if path.is_file() and path.name != Path(__file__).name
        ),
    }
    (OUT / "resumo_execucao.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    print("\nResumo estadual:")
    print(summary.to_string(index=False))
    print("\nModelo principal:")
    print(
        coefficients[
            coefficients["modelo"].eq(
                "Logit condicional — receita total por ano"
            )
        ].to_string(index=False)
    )
    print("\nContrastes:")
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
