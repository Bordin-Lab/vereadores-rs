# Teste do acoplamento entre dinheiro, votos e eleição

Este pacote integra o painel de receitas e despesas das candidaturas a vereador
com os resultados nominais e partidários do TSE para os 497 municípios do Rio
Grande do Sul em 2016, 2020 e 2024. O objetivo é decompor o acoplamento
dinheiro–eleição em três elos observáveis:

1. recursos de campanha associados a votos nominais;
2. votos nominais e força da lista associados à eleição;
3. viabilidade anterior associada simultaneamente a recursos e eleição.

## Resultado principal

Em Pelotas, a correlação de Spearman entre receita e votos nominais é 0,663 em
2016, 0,636 em 2020 e 0,668 em 2024. A associação permanece forte quando os
ranks de receita e votos são comparados apenas dentro da mesma lista eleitoral:
0,621, 0,594 e 0,580, respectivamente.

No conjunto do RS, um modelo linear de probabilidade com efeito fixo municipal
estima que o coeficiente de um dobro da receita cai 74,8% em 2016, 87,2% em
2020 e 93,2% em 2024 após a inclusão dos votos nominais e da força da lista.
Isso é compatível com um mecanismo no qual a associação dinheiro–eleição passa
principalmente pela conversão em votos, mas não identifica mediação causal.

Na validação cruzada de Pelotas, a adição do dinheiro a indicadores
pré-campanha aumenta a AUC média em 0,088 em 2020 e 0,115 em 2024. Já depois de
conhecidos os votos correntes e a força da lista, o dinheiro não melhora a
previsão: a variação média da AUC é -0,005 em 2020 e -0,003 em 2024. Portanto,
a hipótese de seleção prévia de candidaturas fortes explica parte, mas não toda,
da informação contida no dinheiro; depois da votação, o dinheiro se torna
essencialmente redundante como preditor.

Em 2024, Pelotas está no percentil 96 entre os municípios gaúchos para a
correlação receita–votos e no percentil 92 para a associação dentro das listas.
O acoplamento local é, portanto, alto mesmo em comparação com o padrão estadual.

## Como reproduzir

1. Baixe apenas os arquivos do RS dos ZIPs nacionais do TSE:

   `python reproducibilidade/download_tse_results_rs.py --output-dir dados_brutos`

2. Disponibilize o painel de receitas/despesas no caminho indicado pela
   constante `PANEL_PATH` em `build_vote_mediation_analysis.py`.

3. Execute:

   `python build_vote_mediation_analysis.py`

Os arquivos oficiais são obtidos das páginas de resultados do TSE:
[2016](https://dadosabertos.tse.jus.br/dataset/resultados-2016),
[2020](https://dadosabertos.tse.jus.br/dataset/resultados-2020) e
[2024](https://dadosabertos.tse.jus.br/dataset/resultados-2024).

## Estrutura

- `build_vote_mediation_analysis.py`: integração, auditorias, modelos e figuras;
- `reproducibilidade/`: extrator remoto e script de download seletivo;
- `tabelas/`: resultados tabulares e painel candidato a candidato de Pelotas;
- `figuras/`: versões PNG e PDF das quatro figuras;
- `dados/painel_candidatos_votos_rs.csv.gz`: painel estadual completo;
- `metodologia.json`: definições operacionais e cautelas.

## Limitações

- Os resultados são observacionais e não identificam o efeito causal do
  financiamento.
- Votos correntes e força corrente da lista são variáveis pós-campanha.
- A viabilidade anterior usa correspondência exata do nome completo normalizado
  no mesmo município; mudanças de nome político, redes familiares, cargos não
  legislativos e apoio informal podem não ser capturados.
- A força da lista segue a regra aplicável a cada pleito: coligação proporcional
  em 2016, partido em 2020 e federação (quando existente) ou partido em 2024.
- Votos nominais quase separam perfeitamente eleitos de não eleitos. Por isso,
  a atenuação com votos é apresentada por modelo linear de probabilidade e
  validação fora da amostra, não por inferência em logit separado.

