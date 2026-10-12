#!/usr/bin/env python3
"""Validate data invariants and key manuscript results."""
from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tolerance):
        fail(f"Expected {expected}, found {actual}")


def validate_panel(path: Path, expected_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing panel: {path}")
    frame = pd.read_csv(
        path,
        dtype={"codigo_municipio": str, "id_candidato": str},
        low_memory=False,
    )
    if len(frame) != 81_974:
        fail(f"{path.name}: expected 81,974 rows, found {len(frame):,}")
    missing = expected_columns.difference(frame.columns)
    if missing:
        fail(f"{path.name}: missing columns {sorted(missing)}")
    if frame.duplicated(["ano", "id_candidato"]).any():
        fail(f"{path.name}: duplicated candidate-year keys")
    counts = frame.groupby("ano").agg(
        candidates=("id_candidato", "size"),
        elected=("eleito", "sum"),
        municipalities=("codigo_municipio", "nunique"),
    )
    expected = {
        2016: (25_622, 4_910, 497),
        2020: (30_311, 4_903, 497),
        2024: (26_041, 4_903, 497),
    }
    for year, values in expected.items():
        actual = tuple(int(counts.loc[year, col]) for col in ["candidates", "elected", "municipalities"])
        if actual != values:
            fail(f"{path.name}, {year}: expected {values}, found {actual}")
    forbidden = {col for col in frame.columns if "cpf" in col.lower() or "cnpj" in col.lower()}
    if forbidden:
        fail(f"Direct tax-identifier columns present: {sorted(forbidden)}")
    return frame


def main() -> None:
    statewide = validate_panel(
        ROOT / "analysis_statewide" / "dados" / "painel_candidatos_rs.csv.gz",
        {"receita_total_2024", "log2_receita", "ideologia"},
    )
    digital = validate_panel(
        ROOT / "analysis_digital" / "dados" / "painel_candidatos_despesas_digitais.csv.gz",
        {"despesa_total_2024", "despesa_digital_2024", "fracao_digital"},
    )
    votes = validate_panel(
        ROOT / "analysis_votes" / "dados" / "painel_candidatos_votos_rs.csv.gz",
        {"votos_nominais", "lista_id", "votos_validos_lista", "retornante"},
    )
    if not statewide[["ano", "id_candidato"]].equals(digital[["ano", "id_candidato"]]):
        # Equality of order is expected in released snapshots; key-set equality is the substantive check.
        if set(map(tuple, statewide[["ano", "id_candidato"]].to_numpy())) != set(
            map(tuple, digital[["ano", "id_candidato"]].to_numpy())
        ):
            fail("Statewide and digital panels do not contain the same candidate-year keys")
    if set(map(tuple, digital[["ano", "id_candidato"]].to_numpy())) != set(
        map(tuple, votes[["ano", "id_candidato"]].to_numpy())
    ):
        fail("Digital and vote panels do not contain the same candidate-year keys")

    pelotas = pd.read_csv(ROOT / "analysis_votes" / "tabelas" / "resumo_pelotas_dinheiro_votos.csv").set_index("ano")
    expected_pelotas = {
        2016: (337, 0.662687, 0.621276, 0.956751),
        2020: (425, 0.635583, 0.594122, 0.922207),
        2024: (252, 0.667655, 0.579634, 0.920429),
    }
    for year, (n, rho, within, auc) in expected_pelotas.items():
        if int(pelotas.loc[year, "candidatos"]) != n:
            fail(f"Pelotas candidate count mismatch in {year}")
        close(pelotas.loc[year, "rho_receita_votos"], rho)
        close(pelotas.loc[year, "rho_receita_votos_dentro_lista"], within)
        close(pelotas.loc[year, "auc_receita_eleicao"], auc)

    attenuation = pd.read_csv(ROOT / "analysis_votes" / "tabelas" / "atenuacao_coeficiente_dinheiro.csv").set_index("ano")
    close(attenuation.loc[2016, "atenuacao_votos_lista"], 0.748484)
    close(attenuation.loc[2020, "atenuacao_votos_lista"], 0.871561)
    close(attenuation.loc[2024, "atenuacao_votos_lista"], 0.932195)

    diagnostics = pd.read_csv(ROOT / "analysis_digital" / "tabelas" / "pelotas_diagnostico_auc.csv")
    d2024 = diagnostics[diagnostics["ano"].eq(2024)].set_index("indicador")
    close(d2024.loc["Despesa total", "auc"], 0.915481)
    close(d2024.loc["Despesa não digital", "auc"], 0.902082)
    close(d2024.loc["Despesa digital", "auc"], 0.835601)

    mechanisms = pd.read_csv(ROOT / "analysis_digital" / "tabelas" / "mecanismos_pelotas.csv").set_index("ano")
    close(mechanisms.loc[2016, "percentil_ajustado_numero_candidatos"], 0.420523)
    close(mechanisms.loc[2020, "percentil_ajustado_numero_candidatos"], 0.418511)
    close(mechanisms.loc[2024, "percentil_ajustado_numero_candidatos"], 0.482897)

    required = [
        ROOT / "paper" / "manuscript.pdf",
        ROOT / "paper" / "supplementary_information.pdf",
        ROOT / "paper" / "figures" / "figure_6_money_votes_digital_mechanisms.pdf",
        ROOT / "README.md",
        ROOT / "CITATION.cff",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            fail(f"Missing or empty artifact: {path}")

    oversized = [path for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size >= 100 * 1024 * 1024]
    if oversized:
        fail("Files exceed GitHub's 100 MiB limit: " + ", ".join(str(p.relative_to(ROOT)) for p in oversized))

    print("Repository validation passed.")
    print("  candidate-year rows: 81,974")
    print("  municipalities per election: 497")
    print("  key vote, attenuation, digital, and size-adjusted results: verified")
    print("  direct CPF/CNPJ columns: absent")
    print("  files >=100 MiB: absent")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise
