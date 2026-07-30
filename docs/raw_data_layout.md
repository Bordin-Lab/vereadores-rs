# Raw-data layout

The processed snapshots are already included. This layout is required only to rebuild every join and audit from the official source archives.

## Candidate registration and campaign receipts

```text
tse_downloads/
├── candidatos/
│   ├── consulta_cand_2016_RS.csv
│   ├── consulta_cand_2020_RS.csv
│   └── consulta_cand_2024_RS.csv
└── extracted/
    ├── receitas_candidatos_prestacao_contas_final_2016_RS.txt
    ├── receitas_candidatos_2020_RS.csv
    └── receitas_candidatos_2024_RS.csv
```

Run:

```bash
python analysis_statewide/build_statewide_panel.py
```

## Candidate expenses

```text
analysis_digital/dados_brutos/
├── despesas_candidatos_prestacao_contas_final_2016_RS.txt
├── despesas_contratadas_candidatos_2020_RS.csv
└── despesas_contratadas_candidatos_2024_RS.csv
```

Run:

```bash
python analysis_digital/build_digital_spending_analysis.py
python analysis_digital/analyze_pelotas_mechanisms.py
python analysis_digital/analyze_pelotas_expense_categories.py
```

## Candidate and party votes

The selective downloader reads the official national ZIP archives and extracts only Rio Grande do Sul files:

```bash
python analysis_votes/reproducibilidade/download_tse_results_rs.py \
  --output-dir analysis_votes/dados_brutos
```

Expected output:

```text
analysis_votes/dados_brutos/
├── votacao_candidato_munzona_2016_RS.csv
├── votacao_candidato_munzona_2020_RS.csv
├── votacao_candidato_munzona_2024_RS.csv
├── votacao_partido_munzona_2016_RS.csv
├── votacao_partido_munzona_2020_RS.csv
└── votacao_partido_munzona_2024_RS.csv
```

Then run:

```bash
python analysis_votes/build_vote_mediation_analysis.py
```

## Encoding and parsing

TSE files are read using the year-specific delimiter and encoding rules implemented in the scripts. Monetary strings are converted before aggregation, and candidate-source totals are rounded to cents to preserve substantive ties in AUC and rank statistics.
