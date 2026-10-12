# Despesas de campanha, digitalização e eleição de vereadores

Análise das candidaturas válidas a vereador nos 497 municípios do Rio Grande
do Sul em 2016, 2020 e 2024, com ênfase em Pelotas. O painel integra receitas,
resultados eleitorais e despesas oficiais do Portal de Dados Abertos do TSE.

## Pergunta central

O crescimento de propaganda na internet e redes sociais ajuda a explicar o
acoplamento excepcionalmente alto entre dinheiro e eleição observado em
Pelotas?

## Resposta resumida

Ajuda, mas não é suficiente. Pelotas tornou-se muito mais digitalizada que o
conjunto estadual em 2020 e 2024. Em 2024, 90,5% dos eleitos declararam alguma
despesa digital, contra 39,4% dos não eleitos. Ainda assim, retirar todo o gasto
digital reduz a AUC da despesa total somente de 0,915 para 0,902. Em validação
cruzada, adicionar adoção e fração digital ao gasto total não melhora a
predição fora da amostra.

A explicação estrutural mais forte é o tamanho da disputa. Pelotas esteve entre
os 1%–3% dos municípios com mais candidatos por vaga nos três pleitos. A AUC
bruta situa Pelotas nos percentis 96–98 do RS, mas, após ajuste pelo número de
candidatos, seus percentis são 42, 42 e 48. A concentração financeira também
deixa de ser excepcional depois do mesmo ajuste.

O tipo de despesa que mais reproduz o acoplamento é publicidade tradicional:
impressos, adesivos, jornais e revistas, carros de som e correspondência. Sua
AUC foi 0,946, 0,914 e 0,921 em Pelotas. O digital cresce como segundo marcador
em 2024, com AUC 0,836.

## Mecanismos avaliados

- **Tamanho e longa cauda:** muitos candidatos por vaga, com grande número de
  candidaturas pouco financiadas, elevam mecanicamente a concordância entre
  recursos e eleição.
- **Composição digital:** em 2024, qualquer gasto digital está associado a
  razão de chances de 1,36 no RS, a gasto total constante; aumentar a fração
  digital entre os orçamentos não apresenta efeito líquido robusto.
- **Partidos:** comparar candidatos apenas dentro da mesma sigla reduz a AUC de
  Pelotas em 0,075 em 2020, 0,028 em 2024 e 0,003 em 2016. A estrutura partidária
  explica uma parte, mas não elimina a associação individual.
- **Incumbência:** foram identificados por nome 15 incumbentes em 2020 e 17 em
  2024. Excluindo-os, a AUC da receita permanece 0,941 e 0,924,
  respectivamente. Portanto, incumbência não é a explicação principal.
- **Profissionalização:** publicidade tradicional é o marcador mais consistente
  dos vencedores e provavelmente representa uma estratégia ampla de
  visibilidade, alcance territorial e coordenação de campanha.

## Cautela causal

Os resultados são associativos. Recursos podem produzir visibilidade e votos,
mas partidos, doadores e candidatos também podem direcionar recursos às
candidaturas que já parecem viáveis. Para separar melhor essas direções, a
próxima extensão recomendada é integrar votos nominais, força da legenda,
incumbência validada e data de cada despesa.

## Reproduzir

Execute, a partir da raiz do projeto:

```bash
python analise_despesas_digitais/build_digital_spending_analysis.py
python analise_despesas_digitais/analyze_pelotas_mechanisms.py
python analise_despesas_digitais/analyze_pelotas_expense_categories.py
```

Os arquivos de despesas brutas são os recortes do RS extraídos dos pacotes
oficiais do TSE. A definição digital estrita inclui:

- `Criação e inclusão de páginas na internet`;
- `Despesa com Impulsionamento de Conteúdos`.

Produção de rádio, televisão, vídeo, jingles e slogans permanece separada como
`Audiovisual`, porque a rubrica não identifica o canal de veiculação.

## Produtos

- `dados/painel_candidatos_despesas_digitais.csv.gz`: painel candidato a candidato;
- `tabelas/`: auditorias, modelos, testes, decomposições e indicadores municipais;
- `figuras/`: oito figuras em PNG e PDF;
- `metodologia.json`: decisões de harmonização e fontes;
- `resumo_execucao.json`: síntese automatizada da execução.

Fontes oficiais:

- [Prestação de contas eleitorais de 2016](https://dadosabertos.tse.jus.br/dataset/prestacao-de-contas-eleitorais-2016)
- [Prestação de contas eleitorais de 2020](https://dadosabertos.tse.jus.br/dataset/prestacao-de-contas-eleitorais-2020)
- [Prestação de contas eleitorais de 2024](https://dadosabertos.tse.jus.br/dataset/prestacao-de-contas-eleitorais-2024)
