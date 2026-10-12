# Painel municipal de financiamento eleitoral — Rio Grande do Sul

Este pacote amplia o estudo de Pelotas para os **497 municípios do Rio Grande
do Sul** nas eleições municipais de **2016, 2020 e 2024**. A unidade primária é
a candidatura a vereador; os resultados comparáveis são agregados em 1.491
estratos município-ano.

## Principais resultados

- A participação conjunta do Fundo Partidário e do FEFC na receita estadual
  passou de **6,7% (2016)** para **38,2% (2020)** e **65,2% (2024)**.
- A associação entre receita e eleição permaneceu forte, mas diminuiu de modo
  monotônico. No logit condicional por município-ano, a razão de chances por
  unidade de `log2(1 + receita_real/1000)` foi:

  - 2016: **3,68** (IC95% 3,46–3,92);
  - 2020: **2,95** (IC95% 2,77–3,14);
  - 2024: **2,60** (IC95% 2,45–2,76).

- O teste global de igualdade das inclinações rejeita estabilidade temporal:
  `LR = 113,23`, `gl = 2`, `p = 2,58 × 10⁻²⁵`.
- A AUC municipal mediana caiu de **0,753** para **0,671** entre 2016 e 2024;
  a redução pareada ocorreu na maior parte das cidades (`p = 4,45 × 10⁻²⁷`).
- Pelotas permaneceu aproximadamente no **percentil 98 da AUC estadual**,
  indicando que o caso original é atípico em seu forte acoplamento entre
  receita e eleição.

Esses resultados sustentam a expansão do artigo e alteram sua narrativa:
o painel estadual mostra **desacoplamento parcial**, ao passo que Pelotas
isoladamente sugere estabilidade.

## Arquivos

- `Painel_RS_Financiamento_Eleitoral_2016_2024.xlsx`: painel de resultados,
  gráficos, métodos, auditorias e tabelas.
- `dados/painel_candidatos_rs.csv.gz`: 81.974 candidaturas válidas, incluindo
  candidaturas com receita zero.
- `tabelas/`: indicadores município-ano, modelos, contrastes, composição das
  fontes, perfis ideológicos, testes e auditorias.
- `build_statewide_panel.py`: harmonização, indicadores e estimação.
- `metodologia.json`: definições analíticas e origem dos arquivos.

## Reprodução

Execute a partir da raiz do projeto:

```bash
python analise_painel_rs/build_statewide_panel.py
```

Dependências: Python 3.12, NumPy, pandas, SciPy e scikit-learn.

Quando disponíveis, o script usa os arquivos oficiais do TSE nos caminhos
documentados em `metodologia.json`. Na ausência deles, reproduz os resultados a
partir de `dados/painel_candidatos_rs.csv.gz`, incluído neste pacote. Todos os
arquivos de candidaturas e receitas usados são estaduais; portanto, nenhum
município foi selecionado ou excluído por porte.

## Decisões analíticas

- Cargo: vereador.
- Inclusão: resultado final diferente de `#NULO`; candidaturas sem receita
  permanecem no denominador com valor zero.
- Eleito: situação final iniciada por `ELEITO`.
- Valores: totais por fonte fixados em centavos antes dos cálculos e corrigidos
  pelo IPCA para reais de outubro de 2024.
- Receita pública: Fundo Partidário + FEFC, identificados pela fonte declarada.
- Empresas: não há registros do TSE classificados como recursos de pessoas
  jurídicas para vereador no RS em 2016. CNPJs de diretórios partidários e de
  outros candidatos não foram indevidamente tratados como empresas.
- Modelo principal: logit condicional exato por município-ano, com
  erros-padrão clusterizados por município.
- Classificação ideológica: Borges e Vidigal (2023), com equivalências
  históricas e teste de sensibilidade que move PDT, PSB e REDE ao centro.

## Fontes

- [Portal de Dados Abertos do TSE](https://dadosabertos.tse.jus.br/)
- [Borges — apêndice da classificação ideológica dos partidos](https://andreborges.org/wp-content/uploads/2023/11/Ideological-classification-of-Brazilian-parties_appendix.pdf)
- IBGE/SIDRA, IPCA geral.
