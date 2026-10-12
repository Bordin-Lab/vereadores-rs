#!/usr/bin/env python3
"""Reproduce analytical tables and figures from the included processed panels.

This route does not re-download TSE archives. It reruns all statistical models
that enter the manuscript from the three versioned analytical panels. Raw-file
classification/audit tables remain immutable provenance outputs; use the
raw-data scripts documented in README.md to rebuild them from TSE archives.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_statewide() -> None:
    module = load_module(
        "statewide_analysis",
        ROOT / "analysis_statewide" / "build_statewide_panel.py",
    )
    module.main()


def run_digital() -> None:
    module = load_module(
        "digital_analysis",
        ROOT / "analysis_digital" / "build_digital_spending_analysis.py",
    )
    panel = pd.read_csv(
        ROOT / "analysis_digital" / "dados" / "painel_candidatos_despesas_digitais.csv.gz",
        dtype={"codigo_municipio": str, "id_candidato": str},
        low_memory=False,
    )
    panel["codigo_municipio"] = module.normalize_id(panel["codigo_municipio"]).str.zfill(5)
    panel["id_candidato"] = module.normalize_id(panel["id_candidato"])

    indicators = module.add_adjusted_percentiles(module.municipal_indicators(panel))
    state_pelotas = module.state_and_pelotas_summary(panel, indicators)
    coefficients, model_comparisons = module.fit_state_models(panel)
    pelotas_auc, pelotas_groups, pelotas_tests, pelotas_cv = module.pelotas_diagnostics(panel)
    ideology = module.ideology_summary(panel)
    context = pd.read_csv(module.CONTEXT_PATH, dtype={"codigo_municipio": str})
    correlations = module.municipal_correlations(indicators, context)

    outputs = {
        "digital_estado_pelotas.csv": state_pelotas,
        "modelos_digitais_coeficientes.csv": coefficients,
        "modelos_digitais_comparacoes.csv": model_comparisons,
        "pelotas_diagnostico_auc.csv": pelotas_auc,
        "pelotas_eleitos_nao_eleitos.csv": pelotas_groups,
        "pelotas_testes.csv": pelotas_tests,
        "pelotas_validacao_cruzada.csv": pelotas_cv,
        "indicadores_despesa_municipio_ano.csv": indicators,
        "perfil_digital_ideologia.csv": ideology,
        "correlacoes_digitais.csv": correlations,
    }
    for name, frame in outputs.items():
        frame.to_csv(module.TABLE_OUT / name, index=False)

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
    cv_summary.to_csv(module.TABLE_OUT / "pelotas_validacao_cruzada_resumo.csv", index=False)
    module.make_figures(state_pelotas, pelotas_auc, pelotas_groups, coefficients, indicators)
    module.write_methodology()

    mechanisms = load_module(
        "digital_mechanisms",
        ROOT / "analysis_digital" / "analyze_pelotas_mechanisms.py",
    )
    mechanisms.main()
    categories = load_module(
        "expense_categories",
        ROOT / "analysis_digital" / "analyze_pelotas_expense_categories.py",
    )
    categories.main()


def run_votes() -> None:
    module = load_module(
        "vote_analysis",
        ROOT / "analysis_votes" / "build_vote_mediation_analysis.py",
    )
    panel = pd.read_csv(
        ROOT / "analysis_votes" / "dados" / "painel_candidatos_votos_rs.csv.gz",
        dtype={"codigo_municipio": str, "id_candidato": str},
        low_memory=False,
    )
    panel["codigo_municipio"] = module.normalize_id(panel["codigo_municipio"]).str.zfill(5)
    panel["id_candidato"] = module.normalize_id(panel["id_candidato"])

    money_vote = module.money_vote_models(panel)
    coefficients, comparisons = module.fit_election_models(panel)
    lpm_coefficients, attenuation = module.election_lpm_models(panel)
    pelotas = module.pelotas_summary(panel)
    cv_folds, cv_summary, cv_contrasts = module.pelotas_cross_validation(panel)
    cities = module.city_correlations(panel)

    outputs = {
        "modelos_dinheiro_votos.csv": money_vote,
        "modelos_eleicao_coeficientes.csv": coefficients,
        "modelos_eleicao_comparacoes.csv": comparisons,
        "modelos_eleicao_lpm.csv": lpm_coefficients,
        "atenuacao_coeficiente_dinheiro.csv": attenuation,
        "resumo_pelotas_dinheiro_votos.csv": pelotas,
        "validacao_cruzada_pelotas_folds.csv": cv_folds,
        "validacao_cruzada_pelotas_resumo.csv": cv_summary,
        "validacao_cruzada_pelotas_contrastes.csv": cv_contrasts,
        "indicadores_municipio_dinheiro_votos.csv": cities,
    }
    for name, frame in outputs.items():
        frame.to_csv(module.TABLE_OUT / name, index=False)
    panel[panel["codigo_municipio"].eq(module.PELOTAS_CODE)].to_csv(
        module.TABLE_OUT / "painel_candidatos_pelotas_votos.csv", index=False
    )
    module.make_figures(panel, pelotas, lpm_coefficients, attenuation, cv_summary)
    module.write_methodology()


def sync_paper_tables() -> None:
    paper_tables = ROOT / "paper" / "tables"
    result_tables = ROOT / "results" / "tables"
    paper_tables.mkdir(exist_ok=True)
    result_tables.mkdir(parents=True, exist_ok=True)
    selected = [
        ROOT / "analysis_votes" / "tabelas" / "resumo_pelotas_dinheiro_votos.csv",
        ROOT / "analysis_votes" / "tabelas" / "atenuacao_coeficiente_dinheiro.csv",
        ROOT / "analysis_votes" / "tabelas" / "validacao_cruzada_pelotas_resumo.csv",
        ROOT / "analysis_votes" / "tabelas" / "validacao_cruzada_pelotas_contrastes.csv",
        ROOT / "analysis_digital" / "tabelas" / "pelotas_diagnostico_auc.csv",
        ROOT / "analysis_digital" / "tabelas" / "pelotas_validacao_cruzada_resumo.csv",
        ROOT / "analysis_digital" / "tabelas" / "pelotas_categorias_despesa_resumo.csv",
        ROOT / "analysis_digital" / "tabelas" / "mecanismos_pelotas.csv",
    ]
    for source in selected:
        shutil.copy2(source, paper_tables / source.name)
        shutil.copy2(source, result_tables / source.name)


def build_paper_figures() -> None:
    figures = load_module(
        "paper_figures",
        ROOT / "paper" / "code" / "build_figures.py",
    )
    figures.main()
    integrated = load_module(
        "integrated_figure",
        ROOT / "scripts" / "build_integrated_figure.py",
    )
    integrated.build()


def main() -> None:
    print("[1/5] Statewide panel models")
    run_statewide()
    print("[2/5] Digital-spending models")
    run_digital()
    print("[3/5] Vote and list models")
    run_votes()
    print("[4/5] Synchronizing manuscript tables")
    sync_paper_tables()
    print("[5/5] Building publication figures")
    build_paper_figures()
    print("Processed-data reproduction completed.")


if __name__ == "__main__":
    main()
