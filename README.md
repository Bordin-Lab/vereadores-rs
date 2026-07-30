# Campaign finance, votes, and electoral coupling in Rio Grande do Sul

Reproducibility package for the manuscript:

> **Institutional Rewiring and Partial Electoral Decoupling: Campaign Finance, Votes, and Local Heterogeneity across 497 Brazilian Municipalities**

The project analyzes **81,974 city-council candidacies** in all **497 municipalities of Rio Grande do Sul**, Brazil, for the 2016, 2020, and 2024 municipal elections. It connects campaign receipts, expenditure composition, nominal votes, electoral-list strength, municipal context, and election outcomes.

## Main empirical result

The campaign-finance regime became predominantly public and party-mediated, but publicization did not generally reduce candidate-level financial concentration. Revenue remained strongly associated with election, although the within-municipality response weakened over time.

The extended mechanism tests refine this result:

- the revenue–vote association in Pelotas remained strong and stable;
- current votes and electoral-list strength absorb most of the revenue–election coefficient, but this is a post-campaign diagnostic rather than causal mediation;
- money improves prediction beyond previous-election viability, yet adds no information after current votes and list strength are known;
- digital expenditure grew and became common among winners, but does not improve out-of-sample prediction beyond total expenditure;
- Pelotas's extreme raw revenue–election AUC becomes typical after adjustment for its unusually large number of candidates per seat.

The article therefore distinguishes **publicization**, **concentration**, **revenue–vote alignment**, and **revenue–election selection** as separate observables.

## Repository structure

```text
.
├── analysis_statewide/   # statewide receipts, inequality, conditional models, context
├── analysis_digital/     # expenses, digital adoption, categories, Pelotas diagnostics
├── analysis_votes/       # nominal votes, electoral lists, prior viability, attenuation
├── paper/                # LaTeX sources, compiled PDFs, figures, and paper tables
├── results/              # integrated machine-readable outputs
├── scripts/              # reproduction, validation, figure, and compilation utilities
├── tests/                # lightweight automated tests
├── workbooks/            # convenience Excel output; CSV files are authoritative
└── docs/                 # data dictionary, raw-data layout, and release notes
```

## Installation

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate campaign-finance-rs
```

## Reproduction routes

### 1. Validate the released analytical snapshots

This is the fastest integrity check:

```bash
make validate
```

It verifies candidate/election totals, municipality coverage, unique keys, absence of direct CPF/CNPJ columns, GitHub file-size limits, and the key numerical results used in the manuscript.

### 2. Re-run models from the included processed panels

```bash
make reproduce
```

This route re-estimates the statewide, digital, vote, list, and cross-validation models from the released candidate-level analytical panels. The exact conditional-likelihood calculations are computationally intensive and can take several minutes on a laptop.

Then regenerate the integrated figures and PDFs:

```bash
make figures
make paper
```

The paper compiler automatically uses a working BibTeX executable and creates:

- `paper/manuscript.pdf`
- `paper/supplementary_information.pdf`

### 3. Rebuild from official TSE files

The repository does not duplicate the large national raw archives. Place the extracted Rio Grande do Sul files according to [`docs/raw_data_layout.md`](docs/raw_data_layout.md), then run:

```bash
python analysis_statewide/build_statewide_panel.py
python analysis_digital/build_digital_spending_analysis.py
python analysis_digital/analyze_pelotas_mechanisms.py
python analysis_digital/analyze_pelotas_expense_categories.py
python analysis_votes/reproducibilidade/download_tse_results_rs.py \
  --output-dir analysis_votes/dados_brutos
python analysis_votes/build_vote_mediation_analysis.py
python scripts/reproduce_from_processed.py
bash scripts/compile_paper.sh
```

The raw-data route preserves audit tables for joins, excluded records, zero-value transactions, and source/type harmonization.

## Data policy

The candidate-level panels are derived from public TSE administrative records. They contain candidate names and public electoral attributes needed for reproducibility, but **do not contain CPF, CNPJ, addresses, donor tax identifiers, or transaction-level personal identifiers**. Monetary values are expressed both nominally and in October 2024 BRL where required.

CSV files are the authoritative machine-readable outputs. Excel workbooks are provided only for convenient inspection.

## Statistical interpretation

All results are observational. Revenue and viability are jointly determined, and parties may target candidates they already consider competitive. Current votes and list strength are post-campaign variables; attenuation after adding them demonstrates predictive redundancy, not a causal mediation effect. Expense labels describe contracted purposes, not actual audience exposure or persuasion.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff).

## Licensing

- Analysis code: MIT License (`LICENSE-CODE`)
- Derived data, documentation, tables, and original figures: CC BY 4.0 (`LICENSE-DATA`), subject to attribution and the terms of the public source datasets
- Springer Nature template files retain their original licensing terms
