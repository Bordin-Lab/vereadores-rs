"""Dinheiro, votos e eleição de vereadores no Rio Grande do Sul.

Integra votação nominal e partidária do TSE ao painel de receitas/despesas e
decompõe o acoplamento dinheiro--eleição em:

1. associação entre recursos e votos nominais;
2. força da lista eleitoral (coligação em 2016, partido em 2020 e
   partido/federação em 2024);
3. associação mecânica entre votos, lista e eleição;
4. viabilidade prévia observável: retorno, incumbência, votos anteriores e
   força partidária anterior.

Os resultados são associativos. Votos correntes são mediadores pós-campanha,
não controles causais exógenos.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-votos-pelotas")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, spearmanr
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

PANEL_PATH = (
    ROOT
    / "analysis_digital"
    / "dados"
    / "painel_candidatos_despesas_digitais.csv.gz"
)

sys.path.insert(0, str(ROOT / "analysis_statewide"))
from build_statewide_panel import fit_conditional_logit  # noqa: E402


YEARS = [2016, 2020, 2024]
PELOTAS_CODE = "87912"
VOTE_PATHS = {
    year: RAW / f"votacao_candidato_munzona_{year}_RS.csv"
    for year in YEARS
}
PARTY_PATHS = {
    year: RAW / f"votacao_partido_munzona_{year}_RS.csv"
    for year in YEARS
}
YEAR_COLORS = {
    2016: "#0072B2",
    2020: "#E69F00",
    2024: "#009E73",
}
MODEL_COLORS = {
    "Somente dinheiro": "#6A3D9A",
    "Dinheiro + força da lista": "#E69F00",
    "Dinheiro + votos + lista": "#009E73",
}

# Continuidade de siglas entre pleitos. Valores são siglas no pleito anterior.
PREDECESSORS = {
    2020: {
        "MDB": ["PMDB"],
        "REPUBLICANOS": ["PRB"],
        "CIDADANIA": ["PPS"],
        "PODE": ["PTN"],
        "AVANTE": ["PT do B", "PTDO B"],
        "SOLIDARIEDADE": ["SD", "SOLIDARIEDADE"],
        "DC": ["PSDC"],
        "PATRIOTA": ["PEN"],
        "PL": ["PR"],
    },
    2024: {
        "UNIÃO": ["DEM", "PSL"],
        "PRD": ["PTB", "PATRIOTA"],
    },
}


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").upper())
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def normalize_party(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def valid_positive_code(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(-1).gt(0)


def list_id(frame: pd.DataFrame, year: int) -> pd.Series:
    party = frame["SG_PARTIDO"].fillna("#NULO").astype(str).str.strip()
    result = "PARTIDO:" + party
    federation = valid_positive_code(frame["NR_FEDERACAO"])
    result.loc[federation] = (
        "FEDERACAO:" + normalize_id(frame.loc[federation, "NR_FEDERACAO"])
    )
    if year == 2016:
        coalition = (
            valid_positive_code(frame["SQ_COLIGACAO"])
            & frame["TP_AGREMIACAO"]
            .fillna("")
            .str.upper()
            .str.contains("COLIGA", regex=False)
        )
        result.loc[coalition] = (
            "COLIGACAO:" + normalize_id(frame.loc[coalition, "SQ_COLIGACAO"])
        )
    return result


def safe_auc(y, score) -> float:
    y_array = np.asarray(y, dtype=int)
    if len(np.unique(y_array)) < 2:
        return np.nan
    return float(roc_auc_score(y_array, np.asarray(score, dtype=float)))


def safe_spearman(x, y) -> tuple[float, float]:
    valid = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(valid["x"], valid["y"])
    return float(result.statistic), float(result.pvalue)


def load_candidate_votes(year: int) -> tuple[pd.DataFrame, dict]:
    path = VOTE_PATHS[year]
    columns = [
        "ANO_ELEICAO",
        "NR_TURNO",
        "SG_UE",
        "NM_UE",
        "NR_ZONA",
        "DS_CARGO",
        "SQ_CANDIDATO",
        "NM_CANDIDATO",
        "SG_PARTIDO",
        "NR_FEDERACAO",
        "SG_FEDERACAO",
        "DS_COMPOSICAO_FEDERACAO",
        "SQ_COLIGACAO",
        "NM_COLIGACAO",
        "DS_COMPOSICAO_COLIGACAO",
        "TP_AGREMIACAO",
        "QT_VOTOS_NOMINAIS",
        "QT_VOTOS_NOMINAIS_VALIDOS",
        "NM_TIPO_DESTINACAO_VOTOS",
    ]
    frame = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        dtype=str,
        usecols=columns,
        low_memory=False,
    )
    rows_file = len(frame)
    frame = frame[
        frame["DS_CARGO"].fillna("").str.upper().eq("VEREADOR")
        & frame["NR_TURNO"].fillna("").eq("1")
    ].copy()
    rows_council = len(frame)
    frame["ano"] = year
    frame["codigo_municipio"] = normalize_id(frame["SG_UE"]).str.zfill(5)
    frame["id_candidato"] = normalize_id(frame["SQ_CANDIDATO"])
    frame["votos_nominais"] = numeric(frame["QT_VOTOS_NOMINAIS"])
    frame["votos_nominais_validos"] = numeric(
        frame["QT_VOTOS_NOMINAIS_VALIDOS"]
    )
    frame["lista_id"] = list_id(frame, year)
    metadata = [
        "ano",
        "codigo_municipio",
        "id_candidato",
        "NM_UE",
        "NM_CANDIDATO",
        "SG_PARTIDO",
        "NR_FEDERACAO",
        "SG_FEDERACAO",
        "DS_COMPOSICAO_FEDERACAO",
        "SQ_COLIGACAO",
        "NM_COLIGACAO",
        "DS_COMPOSICAO_COLIGACAO",
        "TP_AGREMIACAO",
        "lista_id",
    ]
    aggregate = (
        frame.groupby(
            ["ano", "codigo_municipio", "id_candidato"],
            as_index=False,
        )
        .agg(
            municipio_votos=("NM_UE", "first"),
            nome_candidato_votos=("NM_CANDIDATO", "first"),
            partido_votos=("SG_PARTIDO", "first"),
            nr_federacao=("NR_FEDERACAO", "first"),
            sigla_federacao=("SG_FEDERACAO", "first"),
            composicao_federacao=("DS_COMPOSICAO_FEDERACAO", "first"),
            sq_coligacao=("SQ_COLIGACAO", "first"),
            nome_coligacao=("NM_COLIGACAO", "first"),
            composicao_coligacao=("DS_COMPOSICAO_COLIGACAO", "first"),
            tipo_agremiacao=("TP_AGREMIACAO", "first"),
            lista_id=("lista_id", "first"),
            votos_nominais=("votos_nominais", "sum"),
            votos_nominais_validos=("votos_nominais_validos", "sum"),
            zonas=("NR_ZONA", "nunique"),
        )
    )
    audit = {
        "ano": year,
        "arquivo": path.name,
        "linhas_arquivo": rows_file,
        "linhas_vereador_turno_1": rows_council,
        "candidatos_arquivo_votacao": aggregate["id_candidato"].nunique(),
        "votos_nominais_validos_arquivo": aggregate[
            "votos_nominais_validos"
        ].sum(),
    }
    return aggregate, audit


def load_party_votes(
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = PARTY_PATHS[year]
    columns = [
        "ANO_ELEICAO",
        "NR_TURNO",
        "SG_UE",
        "NM_UE",
        "NR_ZONA",
        "DS_CARGO",
        "TP_AGREMIACAO",
        "SG_PARTIDO",
        "NR_FEDERACAO",
        "SG_FEDERACAO",
        "DS_COMPOSICAO_FEDERACAO",
        "SQ_COLIGACAO",
        "NM_COLIGACAO",
        "DS_COMPOSICAO_COLIGACAO",
        "QT_VOTOS_LEGENDA_VALIDOS",
        "QT_VOTOS_NOM_CONVR_LEG_VALIDOS",
        "QT_TOTAL_VOTOS_LEG_VALIDOS",
        "QT_VOTOS_NOMINAIS_VALIDOS",
    ]
    frame = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        dtype=str,
        usecols=columns,
        low_memory=False,
    )
    rows_file = len(frame)
    frame = frame[
        frame["DS_CARGO"].fillna("").str.upper().eq("VEREADOR")
        & frame["NR_TURNO"].fillna("").eq("1")
    ].copy()
    rows_council = len(frame)
    frame["ano"] = year
    frame["codigo_municipio"] = normalize_id(frame["SG_UE"]).str.zfill(5)
    frame["lista_id"] = list_id(frame, year)
    for column in [
        "QT_VOTOS_LEGENDA_VALIDOS",
        "QT_VOTOS_NOM_CONVR_LEG_VALIDOS",
        "QT_TOTAL_VOTOS_LEG_VALIDOS",
        "QT_VOTOS_NOMINAIS_VALIDOS",
    ]:
        frame[column] = numeric(frame[column])

    party = (
        frame.groupby(
            [
                "ano",
                "codigo_municipio",
                "SG_PARTIDO",
                "lista_id",
            ],
            as_index=False,
        )
        .agg(
            municipio_partido=("NM_UE", "first"),
            votos_legenda_validos=("QT_VOTOS_LEGENDA_VALIDOS", "sum"),
            votos_nominais_convertidos_legenda=(
                "QT_VOTOS_NOM_CONVR_LEG_VALIDOS",
                "sum",
            ),
            total_votos_legenda_validos=(
                "QT_TOTAL_VOTOS_LEG_VALIDOS",
                "sum",
            ),
            votos_nominais_validos_partido=(
                "QT_VOTOS_NOMINAIS_VALIDOS",
                "sum",
            ),
        )
    )
    party["votos_validos_partido"] = (
        party["votos_nominais_validos_partido"]
        + party["total_votos_legenda_validos"]
    )
    party["partido_normalizado"] = party["SG_PARTIDO"].map(normalize_party)

    lists = (
        party.groupby(
            ["ano", "codigo_municipio", "lista_id"],
            as_index=False,
        )
        .agg(
            votos_nominais_validos_lista=(
                "votos_nominais_validos_partido",
                "sum",
            ),
            votos_legenda_validos_lista=(
                "total_votos_legenda_validos",
                "sum",
            ),
            votos_validos_lista=("votos_validos_partido", "sum"),
            partidos_na_lista=("SG_PARTIDO", "nunique"),
        )
    )
    lists["votos_validos_municipio"] = lists.groupby(
        ["ano", "codigo_municipio"]
    )["votos_validos_lista"].transform("sum")
    lists["fracao_votos_lista"] = np.where(
        lists["votos_validos_municipio"].gt(0),
        lists["votos_validos_lista"] / lists["votos_validos_municipio"],
        0.0,
    )
    party_totals = party.groupby(
        ["ano", "codigo_municipio"]
    )["votos_validos_partido"].transform("sum")
    party["fracao_votos_partido"] = np.where(
        party_totals.gt(0),
        party["votos_validos_partido"] / party_totals,
        0.0,
    )
    audit = {
        "ano": year,
        "arquivo_partido": path.name,
        "linhas_partido_arquivo": rows_file,
        "linhas_partido_vereador_turno_1": rows_council,
        "listas_municipio": len(lists),
        "votos_validos_listas": lists["votos_validos_lista"].sum(),
    }
    return party, lists, audit


def current_components(row: pd.Series, year: int) -> list[str]:
    composition = str(row.get("composicao_federacao", "") or "").strip()
    federation = pd.to_numeric(
        pd.Series([row.get("nr_federacao")]), errors="coerce"
    ).fillna(-1).iloc[0]
    if year == 2024 and federation > 0 and composition not in {
        "",
        "#NULO#",
        "#NULO",
    }:
        components = [
            normalize_party(item)
            for item in composition.split("/")
            if normalize_party(item)
        ]
        if components:
            return components
    return [normalize_party(row.get("partido", ""))]


def predecessor_parties(party: str, year: int) -> list[str]:
    normalized = normalize_party(party)
    return [
        normalize_party(value)
        for value in PREDECESSORS.get(year, {}).get(normalized, [normalized])
    ]


def add_prior_viability(
    panel: pd.DataFrame,
    party_table: pd.DataFrame,
) -> pd.DataFrame:
    result = panel.copy()
    result["nome_normalizado"] = result["nome_candidato"].map(normalize_name)
    for column in [
        "retornante",
        "incumbente",
        "votos_anteriores",
        "log2_votos_anteriores",
        "fracao_lista_anterior",
        "fracao_lista_anterior_10pp",
        "lista_anterior_disponivel",
    ]:
        result[column] = 0.0

    party_lookup = {
        (
            int(row.ano),
            str(row.codigo_municipio).zfill(5),
            normalize_party(row.SG_PARTIDO),
        ): float(row.fracao_votos_partido)
        for row in party_table.itertuples(index=False)
    }

    for current_year, previous_year in [(2020, 2016), (2024, 2020)]:
        previous = result[result["ano"].eq(previous_year)][
            [
                "codigo_municipio",
                "nome_normalizado",
                "votos_nominais_validos",
                "eleito",
            ]
        ].copy()
        previous = (
            previous.groupby(
                ["codigo_municipio", "nome_normalizado"],
                as_index=False,
            )
            .agg(
                votos_anteriores=("votos_nominais_validos", "max"),
                incumbente=("eleito", "max"),
            )
        )
        current_mask = result["ano"].eq(current_year)
        current = result.loc[current_mask].copy()
        current["_original_index"] = current.index
        current = current.merge(
            previous,
            on=["codigo_municipio", "nome_normalizado"],
            how="left",
            suffixes=("", "_match"),
        ).set_index("_original_index")
        matched = current["votos_anteriores_match"].notna()
        result.loc[current.index, "retornante"] = matched.astype(int)
        result.loc[current.index, "incumbente"] = (
            current["incumbente_match"].fillna(0).astype(int)
        )
        result.loc[current.index, "votos_anteriores"] = current[
            "votos_anteriores_match"
        ].fillna(0)

        prior_shares = []
        prior_available = []
        for _, row in current.iterrows():
            components = current_components(row, current_year)
            predecessors = {
                predecessor
                for component in components
                for predecessor in predecessor_parties(component, current_year)
            }
            shares = [
                party_lookup[
                    (
                        previous_year,
                        str(row["codigo_municipio"]).zfill(5),
                        predecessor,
                    )
                ]
                for predecessor in predecessors
                if (
                    previous_year,
                    str(row["codigo_municipio"]).zfill(5),
                    predecessor,
                )
                in party_lookup
            ]
            prior_shares.append(float(sum(shares)))
            prior_available.append(int(bool(shares)))
        result.loc[current.index, "fracao_lista_anterior"] = prior_shares
        result.loc[current.index, "lista_anterior_disponivel"] = prior_available

    result["log2_votos_anteriores"] = np.log2(
        1 + result["votos_anteriores"]
    )
    result["fracao_lista_anterior_10pp"] = (
        result["fracao_lista_anterior"] / 0.10
    )
    return result


def integrate_panel() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    candidate_frames = []
    party_frames = []
    list_frames = []
    audits = []
    for year in YEARS:
        candidate, candidate_audit = load_candidate_votes(year)
        party, lists, party_audit = load_party_votes(year)
        candidate_frames.append(candidate)
        party_frames.append(party)
        list_frames.append(lists)
        audits.append({**candidate_audit, **party_audit})

    candidate_votes = pd.concat(candidate_frames, ignore_index=True)
    party_votes = pd.concat(party_frames, ignore_index=True)
    list_votes = pd.concat(list_frames, ignore_index=True)
    panel = pd.read_csv(
        PANEL_PATH,
        dtype={"codigo_municipio": str, "id_candidato": str},
    )
    panel["codigo_municipio"] = normalize_id(
        panel["codigo_municipio"]
    ).str.zfill(5)
    panel["id_candidato"] = normalize_id(panel["id_candidato"])

    merge_columns = [
        "ano",
        "id_candidato",
        "codigo_municipio",
        "partido_votos",
        "nr_federacao",
        "sigla_federacao",
        "composicao_federacao",
        "sq_coligacao",
        "nome_coligacao",
        "composicao_coligacao",
        "tipo_agremiacao",
        "lista_id",
        "votos_nominais",
        "votos_nominais_validos",
        "zonas",
    ]
    panel = panel.merge(
        candidate_votes[merge_columns],
        on=["ano", "id_candidato", "codigo_municipio"],
        how="left",
        validate="one_to_one",
    )
    panel["votos_ausentes_arquivo"] = panel["lista_id"].isna().astype(int)
    panel["votos_nominais"] = panel["votos_nominais"].fillna(0)
    panel["votos_nominais_validos"] = panel[
        "votos_nominais_validos"
    ].fillna(0)
    panel["lista_id"] = panel["lista_id"].fillna(
        "PARTIDO:" + panel["partido"].fillna("#NULO")
    )

    panel = panel.merge(
        list_votes,
        on=["ano", "codigo_municipio", "lista_id"],
        how="left",
        validate="many_to_one",
    )
    for column in [
        "votos_nominais_validos_lista",
        "votos_legenda_validos_lista",
        "votos_validos_lista",
        "votos_validos_municipio",
        "fracao_votos_lista",
        "partidos_na_lista",
    ]:
        panel[column] = panel[column].fillna(0)
    panel["fracao_votos_candidato_municipio"] = np.where(
        panel["votos_validos_municipio"].gt(0),
        panel["votos_nominais_validos"]
        / panel["votos_validos_municipio"],
        0.0,
    )
    panel["fracao_votos_candidato_lista"] = np.where(
        panel["votos_validos_lista"].gt(0),
        panel["votos_nominais_validos"] / panel["votos_validos_lista"],
        0.0,
    )
    panel["fracao_votos_lista_10pp"] = panel["fracao_votos_lista"] / 0.10
    panel["log2_votos"] = np.log2(1 + panel["votos_nominais_validos"])
    panel["log2_votos_lista_milhar"] = np.log2(
        1 + panel["votos_validos_lista"] / 1000
    )
    panel["rank_votos_lista"] = panel.groupby(
        ["ano", "codigo_municipio", "lista_id"]
    )["votos_nominais_validos"].rank(method="min", ascending=False)

    panel = add_prior_viability(panel, party_votes)

    panel_keys = set(zip(panel["ano"], panel["id_candidato"]))
    vote_keys = set(
        zip(candidate_votes["ano"], candidate_votes["id_candidato"])
    )
    for audit in audits:
        year = audit["ano"]
        panel_year = {key for key in panel_keys if key[0] == year}
        vote_year = {key for key in vote_keys if key[0] == year}
        audit["candidatos_painel"] = len(panel_year)
        audit["candidatos_integrados"] = len(panel_year & vote_year)
        audit["candidatos_painel_sem_linha_votacao"] = len(panel_year - vote_year)
        audit["candidatos_votacao_fora_painel"] = len(vote_year - panel_year)
        audit["votos_validos_painel"] = panel.loc[
            panel["ano"].eq(year), "votos_nominais_validos"
        ].sum()
        audit["diferenca_nominais_lista_menos_painel"] = (
            list_votes.loc[
                list_votes["ano"].eq(year),
                "votos_nominais_validos_lista",
            ].sum()
            - audit["votos_validos_painel"]
        )

    return (
        panel,
        party_votes,
        list_votes,
        pd.DataFrame(audits),
    )


def fit_within_ols(
    frame: pd.DataFrame,
    x_columns: list[str],
    y_column: str,
    group_column: str,
    cluster_column: str,
) -> dict:
    model_columns = list(
        dict.fromkeys(x_columns + [y_column, group_column, cluster_column])
    )
    use = frame[model_columns].dropna()
    X = use[x_columns].to_numpy(float)
    y = use[y_column].to_numpy(float)
    groups = use[group_column]
    clusters = use[cluster_column]
    X_tilde = X - use.groupby(group_column)[x_columns].transform(
        "mean"
    ).to_numpy(float)
    y_tilde = y - use.groupby(group_column)[y_column].transform(
        "mean"
    ).to_numpy(float)
    information = X_tilde.T @ X_tilde
    bread = np.linalg.pinv(information)
    beta = bread @ X_tilde.T @ y_tilde
    residual = y_tilde - X_tilde @ beta
    cluster_codes, cluster_labels = pd.factorize(clusters, sort=True)
    scores = np.vstack(
        [
            np.bincount(
                cluster_codes,
                weights=X_tilde[:, parameter] * residual,
                minlength=len(cluster_labels),
            )
            for parameter in range(len(x_columns))
        ]
    ).T
    meat = scores.T @ scores
    correction = (
        len(cluster_labels) / (len(cluster_labels) - 1)
        * (len(use) - 1)
        / (len(use) - len(x_columns))
    )
    covariance = correction * bread @ meat @ bread
    return {
        "beta": beta,
        "standard_error": np.sqrt(np.maximum(np.diag(covariance), 0)),
        "parameter_names": x_columns,
        "n": len(use),
        "groups": groups.nunique(),
        "clusters": len(cluster_labels),
        "r2_within": 1
        - np.sum(residual**2)
        / np.sum((y_tilde - y_tilde.mean()) ** 2),
    }


def money_vote_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        frame = panel[panel["ano"].eq(year)].copy()
        frame["municipio_lista"] = (
            frame["codigo_municipio"] + "|" + frame["lista_id"]
        )
        for money, label in [
            ("log2_receita", "Receita"),
            ("log2_despesa_total", "Despesa"),
        ]:
            for fixed_effect, group, model_label in [
                ("Município", "codigo_municipio", "FE município"),
                ("Lista", "municipio_lista", "FE lista eleitoral"),
            ]:
                model = fit_within_ols(
                    frame,
                    [money],
                    "log2_votos",
                    group,
                    "codigo_municipio",
                )
                beta = model["beta"][0]
                se = model["standard_error"][0]
                z_score = beta / se
                rows.append(
                    {
                        "ano": year,
                        "recurso": label,
                        "efeito_fixo": fixed_effect,
                        "modelo": model_label,
                        "beta_log2_votos": beta,
                        "erro_padrao_cluster": se,
                        "multiplicador_votos_por_dobro_recurso": 2**beta,
                        "ic95_inferior_multiplicador": 2 ** (beta - 1.96 * se),
                        "ic95_superior_multiplicador": 2 ** (beta + 1.96 * se),
                        "p_valor": 2 * norm.sf(abs(z_score)),
                        "r2_within": model["r2_within"],
                        "n": model["n"],
                        "grupos": model["groups"],
                        "municipios": model["clusters"],
                    }
                )
    return pd.DataFrame(rows)


def election_lpm_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate stable descriptive attenuation with municipality fixed effects.

    The exact conditional logit is appropriate before adding current votes, but
    current nominal votes nearly separate elected from non-elected candidates.
    A linear probability model remains estimable and makes the coefficient
    attenuation transparent in probability points.
    """

    specifications = [
        ("Somente dinheiro", ["log2_receita"]),
        (
            "Dinheiro + força da lista",
            ["log2_receita", "fracao_votos_lista_10pp"],
        ),
        (
            "Dinheiro + votos + lista",
            ["log2_receita", "log2_votos", "fracao_votos_lista_10pp"],
        ),
        (
            "Votos + lista",
            ["log2_votos", "fracao_votos_lista_10pp"],
        ),
    ]
    parameter_labels = {
        "log2_receita": "Receita: dobro",
        "log2_votos": "Votos nominais: dobro",
        "fracao_votos_lista_10pp": "Força da lista: +10 p.p.",
    }
    rows = []
    attenuation_rows = []
    for year in YEARS:
        frame = panel[panel["ano"].eq(year)].copy()
        money_betas = {}
        for model_name, variables in specifications:
            model = fit_within_ols(
                frame,
                variables,
                "eleito",
                "codigo_municipio",
                "codigo_municipio",
            )
            for variable, beta, se in zip(
                model["parameter_names"],
                model["beta"],
                model["standard_error"],
            ):
                z_score = beta / se if se > 0 else np.nan
                rows.append(
                    {
                        "ano": year,
                        "modelo": model_name,
                        "parametro": parameter_labels[variable],
                        "beta_probabilidade": beta,
                        "beta_pontos_percentuais": 100 * beta,
                        "erro_padrao_pontos_percentuais": 100 * se,
                        "ic95_inferior_pontos_percentuais": 100
                        * (beta - 1.96 * se),
                        "ic95_superior_pontos_percentuais": 100
                        * (beta + 1.96 * se),
                        "p_valor": 2 * norm.sf(abs(z_score)),
                        "r2_within": model["r2_within"],
                        "n": model["n"],
                        "municipios": model["clusters"],
                    }
                )
                if variable == "log2_receita":
                    money_betas[model_name] = beta
        baseline = money_betas["Somente dinheiro"]
        with_list = money_betas["Dinheiro + força da lista"]
        with_votes = money_betas["Dinheiro + votos + lista"]
        attenuation_rows.append(
            {
                "ano": year,
                "metodo": "LPM com efeito fixo municipal",
                "beta_dinheiro_sem_votos_pp": 100 * baseline,
                "beta_dinheiro_com_lista_pp": 100 * with_list,
                "beta_dinheiro_com_votos_lista_pp": 100 * with_votes,
                "atenuacao_lista": 1 - with_list / baseline,
                "atenuacao_votos_lista": 1 - with_votes / baseline,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(attenuation_rows)


def conditional_rows(
    model: dict,
    year: int,
    model_name: str,
    stage: str,
) -> pd.DataFrame:
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
                "etapa": stage,
                "modelo": model_name,
                "parametro": parameter,
                "beta": beta,
                "erro_padrao_cluster": se,
                "odds_ratio": math.exp(beta),
                "or_ic95_inferior": math.exp(beta - 1.96 * se),
                "or_ic95_superior": math.exp(beta + 1.96 * se),
                "p_valor": 2 * norm.sf(abs(z_score)),
                "log_verossimilhanca": -model["negative_log_likelihood"],
                "n": model["n"],
                "municipios": model["n_groups"],
                "convergiu": model["success"],
                "gradiente_maximo": model["max_abs_gradient"],
            }
        )
    return pd.DataFrame(rows)


def fit_election_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the exact conditional models that remain numerically identified.

    Current nominal votes are intentionally excluded here because they almost
    perfectly separate the elected outcome. Their contribution is evaluated
    with the LPM attenuation and out-of-sample AUC instead.
    """

    coefficient_tables = []
    comparisons = []
    for year in YEARS:
        frame = panel[panel["ano"].eq(year)].copy()
        groups = frame["codigo_municipio"]
        y = frame["eleito"].to_numpy(float)
        money = frame["log2_receita"].to_numpy(float)
        list_share = frame["fracao_votos_lista_10pp"].to_numpy(float)

        model_money = fit_conditional_logit(
            money[:, None],
            y,
            groups,
            groups,
            ["Receita: dobro"],
            np.zeros(1),
        )
        coefficient_tables.append(
            conditional_rows(
                model_money,
                year,
                "Somente dinheiro",
                "Votos correntes",
            )
        )

        model_list = fit_conditional_logit(
            np.column_stack([money, list_share]),
            y,
            groups,
            groups,
            ["Receita: dobro", "Força da lista: +10 p.p."],
            np.r_[model_money["beta"], 0.0],
        )
        coefficient_tables.append(
            conditional_rows(
                model_list,
                year,
                "Dinheiro + força da lista",
                "Votos correntes",
            )
        )

        lr = 2 * (
            model_money["negative_log_likelihood"]
            - model_list["negative_log_likelihood"]
        )
        comparisons.append(
            {
                "ano": year,
                "etapa": "Votos correntes",
                "comparacao": "Dinheiro + lista vs. dinheiro",
                "lr": lr,
                "graus_liberdade": 1,
                "p_valor_lr": chi2.sf(lr, 1),
            }
        )

        if year >= 2020:
            prior = frame[
                [
                    "fracao_lista_anterior_10pp",
                    "lista_anterior_disponivel",
                    "retornante",
                    "log2_votos_anteriores",
                ]
            ].to_numpy(float)
            prior_names = [
                "Força anterior da lista: +10 p.p.",
                "Lista anterior observada",
                "Retornante",
                "Votos anteriores: dobro",
            ]
            model_prior = fit_conditional_logit(
                prior,
                y,
                groups,
                groups,
                prior_names,
                np.zeros(len(prior_names)),
            )
            coefficient_tables.append(
                conditional_rows(
                    model_prior,
                    year,
                    "Viabilidade prévia",
                    "Pré-campanha",
                )
            )
            model_prior_money = fit_conditional_logit(
                np.column_stack([prior, money]),
                y,
                groups,
                groups,
                prior_names + ["Receita: dobro"],
                np.r_[model_prior["beta"], 1.0],
            )
            coefficient_tables.append(
                conditional_rows(
                    model_prior_money,
                    year,
                    "Viabilidade prévia + dinheiro",
                    "Pré-campanha",
                )
            )
            lr = 2 * (
                model_prior["negative_log_likelihood"]
                - model_prior_money["negative_log_likelihood"]
            )
            comparisons.append(
                {
                    "ano": year,
                    "etapa": "Pré-campanha",
                    "comparacao": "Viabilidade prévia + dinheiro vs. prévia",
                    "lr": lr,
                    "graus_liberdade": 1,
                    "p_valor_lr": chi2.sf(lr, 1),
                }
            )

    return (
        pd.concat(coefficient_tables, ignore_index=True),
        pd.DataFrame(comparisons),
    )


def within_list_rank_correlation(frame: pd.DataFrame) -> tuple[float, float]:
    valid = frame[
        ["codigo_municipio", "lista_id", "receita_total_2024", "votos_nominais_validos"]
    ].copy()
    valid["rank_money"] = valid.groupby(
        ["codigo_municipio", "lista_id"]
    )["receita_total_2024"].rank(pct=True, method="average")
    valid["rank_votes"] = valid.groupby(
        ["codigo_municipio", "lista_id"]
    )["votos_nominais_validos"].rank(pct=True, method="average")
    informative = valid.groupby(
        ["codigo_municipio", "lista_id"]
    )["rank_votes"].transform("size").gt(1)
    return safe_spearman(
        valid.loc[informative, "rank_money"],
        valid.loc[informative, "rank_votes"],
    )


def pelotas_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        frame = panel[
            panel["ano"].eq(year)
            & panel["codigo_municipio"].eq(PELOTAS_CODE)
        ].copy()
        revenue_rho, revenue_p = safe_spearman(
            frame["receita_total_2024"], frame["votos_nominais_validos"]
        )
        expense_rho, expense_p = safe_spearman(
            frame["despesa_total_2024"], frame["votos_nominais_validos"]
        )
        within_rho, within_p = within_list_rank_correlation(frame)
        row = {
            "ano": year,
            "candidatos": len(frame),
            "eleitos": int(frame["eleito"].sum()),
            "votos_validos_municipio": frame[
                "votos_validos_municipio"
            ].max(),
            "rho_receita_votos": revenue_rho,
            "p_receita_votos": revenue_p,
            "rho_despesa_votos": expense_rho,
            "p_despesa_votos": expense_p,
            "rho_receita_votos_dentro_lista": within_rho,
            "p_receita_votos_dentro_lista": within_p,
            "auc_receita_eleicao": safe_auc(
                frame["eleito"], frame["receita_total_2024"]
            ),
            "auc_despesa_eleicao": safe_auc(
                frame["eleito"], frame["despesa_total_2024"]
            ),
            "auc_votos_eleicao": safe_auc(
                frame["eleito"], frame["votos_nominais_validos"]
            ),
            "auc_forca_lista_eleicao": safe_auc(
                frame["eleito"], frame["fracao_votos_lista"]
            ),
            "retornantes": int(frame["retornante"].sum()),
            "incumbentes": int(frame["incumbente"].sum()),
            "auc_votos_anteriores_eleicao": (
                safe_auc(frame["eleito"], frame["votos_anteriores"])
                if year >= 2020
                else np.nan
            ),
            "auc_forca_anterior_lista_eleicao": (
                safe_auc(frame["eleito"], frame["fracao_lista_anterior"])
                if year >= 2020
                else np.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def pelotas_cross_validation(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for year in YEARS:
        frame = panel[
            panel["ano"].eq(year)
            & panel["codigo_municipio"].eq(PELOTAS_CODE)
        ].copy()
        feature_sets = {
            "Somente dinheiro": ["log2_receita"],
            "Votos correntes + lista": [
                "log2_votos",
                "fracao_votos_lista_10pp",
            ],
            "Votos correntes + lista + dinheiro": [
                "log2_votos",
                "fracao_votos_lista_10pp",
                "log2_receita",
            ],
        }
        if year >= 2020:
            feature_sets.update(
                {
                    "Viabilidade prévia": [
                        "fracao_lista_anterior_10pp",
                        "lista_anterior_disponivel",
                        "retornante",
                        "log2_votos_anteriores",
                    ],
                    "Viabilidade prévia + dinheiro": [
                        "fracao_lista_anterior_10pp",
                        "lista_anterior_disponivel",
                        "retornante",
                        "log2_votos_anteriores",
                        "log2_receita",
                    ],
                }
            )
        splitter = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=10,
            random_state=year,
        )
        for fold, (train, test) in enumerate(
            splitter.split(frame, frame["eleito"]),
            start=1,
        ):
            for model_name, features in feature_sets.items():
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=year,
                    ),
                )
                model.fit(
                    frame.iloc[train][features],
                    frame.iloc[train]["eleito"],
                )
                probability = model.predict_proba(
                    frame.iloc[test][features]
                )[:, 1]
                fold_rows.append(
                    {
                        "ano": year,
                        "fold": fold,
                        "modelo": model_name,
                        "auc": safe_auc(
                            frame.iloc[test]["eleito"], probability
                        ),
                    }
                )
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["ano", "modelo"], as_index=False)
        .agg(
            auc_media=("auc", "mean"),
            auc_desvio=("auc", "std"),
            auc_p10=("auc", lambda values: values.quantile(0.10)),
            auc_p90=("auc", lambda values: values.quantile(0.90)),
            folds=("auc", "size"),
        )
    )
    contrast_rows = []
    comparisons = [
        (
            "Dinheiro após votos correntes + lista",
            "Votos correntes + lista + dinheiro",
            "Votos correntes + lista",
        ),
        (
            "Dinheiro após viabilidade prévia",
            "Viabilidade prévia + dinheiro",
            "Viabilidade prévia",
        ),
    ]
    for year in YEARS:
        year_folds = folds[folds["ano"].eq(year)]
        for label, expanded_name, base_name in comparisons:
            expanded = year_folds[
                year_folds["modelo"].eq(expanded_name)
            ][["fold", "auc"]].rename(columns={"auc": "auc_expandido"})
            base = year_folds[
                year_folds["modelo"].eq(base_name)
            ][["fold", "auc"]].rename(columns={"auc": "auc_base"})
            paired = base.merge(expanded, on="fold", how="inner")
            if paired.empty:
                continue
            difference = paired["auc_expandido"] - paired["auc_base"]
            contrast_rows.append(
                {
                    "ano": year,
                    "contraste": label,
                    "modelo_base": base_name,
                    "modelo_expandido": expanded_name,
                    "delta_auc_medio": difference.mean(),
                    "delta_auc_mediano": difference.median(),
                    "delta_auc_p10": difference.quantile(0.10),
                    "delta_auc_p90": difference.quantile(0.90),
                    "fracao_folds_delta_positivo": difference.gt(0).mean(),
                    "folds_pareados": len(difference),
                }
            )
    return folds, summary, pd.DataFrame(contrast_rows)


def city_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, code, municipality), frame in panel.groupby(
        ["ano", "codigo_municipio", "municipio"],
        sort=True,
    ):
        rho, p_value = safe_spearman(
            frame["receita_total_2024"], frame["votos_nominais_validos"]
        )
        within_rho, within_p = within_list_rank_correlation(frame)
        rows.append(
            {
                "ano": year,
                "codigo_municipio": code,
                "municipio": municipality,
                "candidatos": len(frame),
                "rho_receita_votos": rho,
                "p_receita_votos": p_value,
                "rho_receita_votos_dentro_lista": within_rho,
                "p_receita_votos_dentro_lista": within_p,
                "auc_receita_eleicao": safe_auc(
                    frame["eleito"], frame["receita_total_2024"]
                ),
                "auc_votos_eleicao": safe_auc(
                    frame["eleito"], frame["votos_nominais_validos"]
                ),
                "auc_forca_lista_eleicao": safe_auc(
                    frame["eleito"], frame["fracao_votos_lista"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    for year in YEARS:
        mask = result["ano"].eq(year)
        for metric in [
            "rho_receita_votos",
            "rho_receita_votos_dentro_lista",
            "auc_votos_eleicao",
        ]:
            result.loc[mask, f"percentil_{metric}"] = result.loc[
                mask, metric
            ].rank(pct=True)
    return result


def make_figures(
    panel: pd.DataFrame,
    pelotas: pd.DataFrame,
    lpm_coefficients: pd.DataFrame,
    attenuation: pd.DataFrame,
    cv_summary: pd.DataFrame,
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

    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.0), sharex=False, sharey=True)
    for axis, year in zip(axes, YEARS):
        frame = panel[
            panel["ano"].eq(year)
            & panel["codigo_municipio"].eq(PELOTAS_CODE)
        ]
        for elected, label, marker, color in [
            (0, "Não eleitos", "o", "#B3B3B3"),
            (1, "Eleitos", "*", YEAR_COLORS[year]),
        ]:
            subset = frame[frame["eleito"].eq(elected)]
            axis.scatter(
                np.log2(1 + subset["receita_total_2024"] / 1000),
                subset["log2_votos"],
                s=55 if elected else 14,
                alpha=0.85 if elected else 0.50,
                marker=marker,
                color=color,
                edgecolor="none",
                label=label,
            )
        axis.set_title(str(year))
        axis.set_xlabel("log₂(1 + receita/1.000)")
        axis.grid(color="#EAEAEA", linewidth=0.7)
        axis.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("log₂(1 + votos nominais)")
    fig.suptitle(
        "Pelotas: recursos e votos nominais caminham juntos",
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_1_dinheiro_votos_pelotas.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    metrics = [
        ("Receita", "auc_receita_eleicao"),
        ("Despesa", "auc_despesa_eleicao"),
        ("Votos nominais", "auc_votos_eleicao"),
        ("Força da lista", "auc_forca_lista_eleicao"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.1, 4.6))
    for index, year in enumerate(YEARS):
        row = pelotas[pelotas["ano"].eq(year)].iloc[0]
        ax.bar(
            x + (index - 1) * width,
            [row[column] for _, column in metrics],
            width=width,
            color=YEAR_COLORS[year],
            label=str(year),
        )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x, [label for label, _ in metrics])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUC para eleição")
    ax.set_title(
        "Pelotas: votos explicam melhor a eleição que o dinheiro",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.legend(title="Ano", frameon=False, ncol=3)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_2_auc_dinheiro_votos_lista.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    money_rows = lpm_coefficients[
        lpm_coefficients["parametro"].eq("Receita: dobro")
        & lpm_coefficients["modelo"].isin(MODEL_COLORS)
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), sharey=True)
    for axis, year in zip(axes, YEARS):
        frame = money_rows[money_rows["ano"].eq(year)]
        positions = np.arange(len(frame))
        for position, row in zip(positions, frame.itertuples(index=False)):
            axis.errorbar(
                row.beta_pontos_percentuais,
                position,
                xerr=np.array(
                    [
                        [
                            row.beta_pontos_percentuais
                            - row.ic95_inferior_pontos_percentuais
                        ],
                        [
                            row.ic95_superior_pontos_percentuais
                            - row.beta_pontos_percentuais
                        ],
                    ]
                ),
                fmt="o",
                color=MODEL_COLORS[row.modelo],
                ecolor=MODEL_COLORS[row.modelo],
                capsize=3,
            )
        axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_yticks(positions, frame["modelo"])
        axis.set_title(str(year))
        axis.set_xlabel("Mudança na probabilidade (p.p.)\npor dobro da receita")
        axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    fig.suptitle(
        "RS: o coeficiente do dinheiro cai após introduzir votos e lista",
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_3_atenuacao_dinheiro.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    plot_cv = cv_summary[cv_summary["ano"].ge(2020)].copy()
    order = [
        "Viabilidade prévia",
        "Somente dinheiro",
        "Viabilidade prévia + dinheiro",
        "Votos correntes + lista",
        "Votos correntes + lista + dinheiro",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for axis, year in zip(axes, [2020, 2024]):
        frame = (
            plot_cv[plot_cv["ano"].eq(year)]
            .set_index("modelo")
            .reindex(order)
        )
        axis.barh(
            np.arange(len(order)),
            frame["auc_media"],
            color=[
                "#8A98A8",
                "#6A3D9A",
                "#E69F00",
                "#009E73",
                "#56B4E9",
            ],
        )
        axis.set_yticks(np.arange(len(order)), order)
        axis.set_xlim(0.45, 1.0)
        axis.set_xlabel("AUC média — validação cruzada")
        axis.set_title(str(year))
        axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    fig.suptitle(
        "Pelotas: viabilidade prévia, dinheiro e informação pós-campanha",
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_OUT / f"figura_4_validacao_viabilidade.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def write_methodology() -> None:
    methodology = {
        "universo": (
            "candidaturas válidas a vereador nos 497 municípios do RS, "
            "2016, 2020 e 2024"
        ),
        "votos_nominais": (
            "soma de QT_VOTOS_NOMINAIS_VALIDOS por SQ_CANDIDATO nas zonas "
            "do município"
        ),
        "forca_lista": {
            "2016": "votos válidos da coligação proporcional",
            "2020": "votos válidos do partido",
            "2024": "votos válidos da federação, quando existente; caso contrário, partido",
            "formula": (
                "votos nominais válidos + QT_TOTAL_VOTOS_LEG_VALIDOS, "
                "divididos pelos votos válidos de todas as listas do município"
            ),
        },
        "viabilidade_previa": (
            "retorno e incumbência por nome completo normalizado no mesmo "
            "município; votos nominais anteriores; participação anterior "
            "do partido ou dos componentes da federação"
        ),
        "modelo_dinheiro_votos": (
            "OLS em log2(1+votos), com efeitos fixos de município ou da lista "
            "eleitoral e erros-padrão clusterizados por município"
        ),
        "modelo_eleicao": (
            "logit condicional exato por município para dinheiro, lista e "
            "viabilidade prévia; LPM com efeito fixo municipal e erros-padrão "
            "clusterizados para a atenuação após votos correntes"
        ),
        "separacao": (
            "votos nominais correntes quase separam perfeitamente eleitos e "
            "não eleitos. Por isso, o logit exato com votos não é usado para "
            "inferência; a contribuição dos votos é avaliada por LPM e AUC "
            "fora da amostra"
        ),
        "cautela": (
            "votos correntes e força corrente da lista são mediadores "
            "pós-campanha. Atenuação do coeficiente não identifica mediação causal."
        ),
        "fontes": {
            "2016": "https://dadosabertos.tse.jus.br/dataset/resultados-2016",
            "2020": "https://dadosabertos.tse.jus.br/dataset/resultados-2020",
            "2024": "https://dadosabertos.tse.jus.br/dataset/resultados-2024",
        },
    }
    (OUT / "metodologia.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    panel, party_votes, list_votes, audit = integrate_panel()
    money_vote = money_vote_models(panel)
    coefficients, comparisons = fit_election_models(panel)
    lpm_coefficients, attenuation = election_lpm_models(panel)
    pelotas = pelotas_summary(panel)
    cv_folds, cv_summary, cv_contrasts = pelotas_cross_validation(panel)
    cities = city_correlations(panel)

    pelotas_panel = panel[
        panel["codigo_municipio"].eq(PELOTAS_CODE)
    ].copy()
    panel.to_csv(DATA_OUT / "painel_candidatos_votos_rs.csv.gz", index=False, compression="gzip")
    pelotas_panel.to_csv(
        TABLE_OUT / "painel_candidatos_pelotas_votos.csv",
        index=False,
    )
    party_votes.to_csv(TABLE_OUT / "votacao_partidos_rs.csv", index=False)
    list_votes.to_csv(TABLE_OUT / "forca_listas_rs.csv", index=False)
    audit.to_csv(TABLE_OUT / "auditoria_votos.csv", index=False)
    money_vote.to_csv(
        TABLE_OUT / "modelos_dinheiro_votos.csv",
        index=False,
    )
    coefficients.to_csv(
        TABLE_OUT / "modelos_eleicao_coeficientes.csv",
        index=False,
    )
    comparisons.to_csv(
        TABLE_OUT / "modelos_eleicao_comparacoes.csv",
        index=False,
    )
    lpm_coefficients.to_csv(
        TABLE_OUT / "modelos_eleicao_lpm.csv",
        index=False,
    )
    attenuation.to_csv(
        TABLE_OUT / "atenuacao_coeficiente_dinheiro.csv",
        index=False,
    )
    pelotas.to_csv(
        TABLE_OUT / "resumo_pelotas_dinheiro_votos.csv",
        index=False,
    )
    cv_folds.to_csv(
        TABLE_OUT / "validacao_cruzada_pelotas_folds.csv",
        index=False,
    )
    cv_summary.to_csv(
        TABLE_OUT / "validacao_cruzada_pelotas_resumo.csv",
        index=False,
    )
    cv_contrasts.to_csv(
        TABLE_OUT / "validacao_cruzada_pelotas_contrastes.csv",
        index=False,
    )
    cities.to_csv(
        TABLE_OUT / "indicadores_municipio_dinheiro_votos.csv",
        index=False,
    )
    make_figures(
        panel,
        pelotas,
        lpm_coefficients,
        attenuation,
        cv_summary,
    )
    write_methodology()

    pelotas_2024 = pelotas[pelotas["ano"].eq(2024)].iloc[0]
    attenuation_2024 = attenuation[attenuation["ano"].eq(2024)].iloc[0]
    contrasts_2024 = cv_contrasts[
        cv_contrasts["ano"].eq(2024)
    ].set_index("contraste")
    summary = {
        "painel_candidatos": len(panel),
        "municipios": panel["codigo_municipio"].nunique(),
        "pelotas_2024": {
            "rho_receita_votos": pelotas_2024["rho_receita_votos"],
            "auc_receita_eleicao": pelotas_2024["auc_receita_eleicao"],
            "auc_votos_eleicao": pelotas_2024["auc_votos_eleicao"],
            "auc_forca_lista_eleicao": pelotas_2024[
                "auc_forca_lista_eleicao"
            ],
            "atenuacao_beta_dinheiro_apos_votos_lista_rs": attenuation_2024[
                "atenuacao_votos_lista"
            ],
            "delta_auc_dinheiro_apos_viabilidade_previa": contrasts_2024.loc[
                "Dinheiro após viabilidade prévia", "delta_auc_medio"
            ],
            "delta_auc_dinheiro_apos_votos_lista": contrasts_2024.loc[
                "Dinheiro após votos correntes + lista", "delta_auc_medio"
            ],
        },
        "interpretacao": (
            "A receita está fortemente associada aos votos. No LPM estadual, "
            "votos nominais e força da lista absorvem a maior parte do "
            "coeficiente do dinheiro. Em Pelotas, dinheiro acrescenta previsão "
            "à viabilidade prévia, mas acrescenta pouco depois de observados "
            "os votos correntes e a lista."
        ),
    }
    (OUT / "resumo_execucao.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
