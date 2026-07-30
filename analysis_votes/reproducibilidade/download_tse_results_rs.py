#!/usr/bin/env python3
"""Baixa seletivamente os arquivos de votação do RS dos ZIPs nacionais do TSE."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from remote_zip_extract import central_directory, extract_member


YEARS = (2016, 2020, 2024)
KINDS = ("votacao_candidato_munzona", "votacao_partido_munzona")
URL_TEMPLATE = (
    "https://cdn.tse.jus.br/estatistica/sead/odsele/"
    "{kind}/{kind}_{year}.zip"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        for kind in KINDS:
            url = URL_TEMPLATE.format(kind=kind, year=year)
            pattern = re.compile(
                rf"(?:^|/){re.escape(kind)}_{year}_RS\.csv$",
                re.IGNORECASE,
            )
            _, entries = central_directory(url)
            selected = [
                entry
                for entry in entries
                if pattern.search(str(entry["name"]))
            ]
            if len(selected) != 1:
                raise RuntimeError(
                    f"Esperado 1 arquivo RS em {url}; encontrados {len(selected)}"
                )
            target = args.output_dir / Path(str(selected[0]["name"])).name
            print(f"{url} -> {target}")
            extract_member(url, selected[0], target)


if __name__ == "__main__":
    main()

