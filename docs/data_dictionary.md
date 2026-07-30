# Data dictionary

## `analysis_statewide/dados/painel_candidatos_rs.csv.gz`

One row per valid city-council candidacy.

| Variable group | Main fields | Meaning |
|---|---|---|
| Keys | `ano`, `codigo_municipio`, `id_candidato` | Election year, TSE municipality code, sequential candidacy ID |
| Candidate | `nome_candidato`, `nome_urna`, `partido`, demographic fields | Public registration attributes |
| Outcome | `situacao_final`, `eleito` | Final TSE status and binary elected indicator |
| Ideology | `ideologia`, `ideologia_sensibilidade` | Primary and sensitivity party-bloc classifications |
| Revenue sources | `Pessoas físicas`, `Recursos próprios`, `Fundo Partidário`, `FEFC`, `Transferências políticas`, `Financiamento coletivo`, `Empresas`, `Outros` | Nominal candidate revenue by harmonized source |
| Revenue transforms | `receita_total_2024`, `log2_receita`, source fields ending in `_2024` | Inflation-adjusted October 2024 BRL and transformed scale |

`Empresas` is an operational source category, not a relabeling of CNPJ records. Direct corporate donations were already prohibited for these elections.

## `analysis_digital/dados/painel_candidatos_despesas_digitais.csv.gz`

Extends the statewide panel with:

- `despesa_total_2024`: total inflation-adjusted expenditure;
- `despesa_digital_2024`: strict digital expenditure;
- `despesa_nao_digital_2024`: total minus strict digital;
- `despesa_impulsionamento_2024`: content boosting;
- `despesa_paginas_internet_2024`: creation/inclusion of internet pages;
- `despesa_audiovisual_2024`: audiovisual production, kept outside strict digital;
- `fracao_digital`, `adotou_digital`: digital share and positive-adoption indicator.

## `analysis_votes/dados/painel_candidatos_votos_rs.csv.gz`

Extends the digital panel with:

- `votos_nominais`: candidate nominal votes;
- `lista_id`: proportional electoral grouping used in that year;
- `votos_validos_lista`: valid votes of the candidate's list;
- `fracao_votos_lista`: list share of municipal valid votes;
- `rank_votos_lista`: within-list vote rank;
- `retornante`, `incumbente`: conservative exact-name indicators;
- `votos_anteriores`, `fracao_lista_anterior`: prior-election viability coordinates.

## Municipal and result tables

- `indicadores_municipio_ano.csv`: Gini, entropy, effective participation, public share, and revenue–election AUC by municipality-year.
- `contexto_municipal.csv`: region, population, urbanization, GDP per capita, and IDHM.
- `mecanismos_pelotas.csv`: candidate/seat scale, raw and adjusted AUC/Gini percentiles, within-party and incumbency diagnostics.
- `atenuacao_coeficiente_dinheiro.csv`: fixed-effect linear-probability attenuation after list strength and votes.
- `pelotas_categorias_despesa_resumo.csv`: expense-category shares, adoption, medians, and election AUC.
