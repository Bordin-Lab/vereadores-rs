# Resumo da atualização - artigo sobre financiamento eleitoral

## Versão científica

O manuscrito foi ampliado do estudo centrado em Pelotas para um painel com 81.974 candidaturas a vereador nos 497 municípios do Rio Grande do Sul, nas eleições de 2016, 2020 e 2024.

A interpretação central passou a distinguir quatro observáveis:

1. publicização das fontes de financiamento;
2. concentração dos recursos entre candidaturas;
3. associação entre receita e votos nominais;
4. separação entre eleitos e não eleitos produzida pela receita.

O resultado principal é descrito como **reconfiguração institucional com desacoplamento eleitoral parcial**. A associação receita-eleição enfraqueceu no município típico, mas permaneceu forte. Ao mesmo tempo, a associação receita-votos permaneceu elevada em Pelotas. A inclusão de votos correntes e força da lista absorve grande parte do coeficiente do dinheiro, mas esse resultado é tratado como decomposição pós-campanha, e não como mediação causal.

Também foram incorporados testes de despesas digitais. A adoção digital é mais frequente entre eleitos, porém não melhora a previsão fora da amostra depois que o gasto total é conhecido. O caráter extremo do AUC bruto de Pelotas é explicado em grande parte pelo tamanho incomum de seu campo competitivo, medido pelo número de candidatos por vaga.

Foram removidas da narrativa principal as interpretações baseadas em árvore de decisão e Lei de Benford, pois não respondem adequadamente à hipótese central e podem induzir conclusões excessivas.

## Reprodutibilidade

O pacote contém:

- manuscrito e informação suplementar em LaTeX e PDF;
- painéis candidato a candidato em CSV compactado;
- tabelas que sustentam os resultados e figuras;
- três planilhas de inspeção em Excel;
- scripts para o painel estadual, despesas digitais e votos/listas;
- ambientes Conda e pip;
- Makefile, testes automatizados e workflow do GitHub Actions;
- dicionário de dados, instruções para os arquivos brutos do TSE, licenças e CITATION.cff;
- manifesto SHA-256 de todos os arquivos.

## Verificações executadas

Foram executados com sucesso:

- compilação sintática dos scripts Python;
- geração das figuras do artigo;
- compilação do manuscrito e da informação suplementar;
- inspeção visual dos PDFs renderizados;
- validação de 81.974 linhas, 497 municípios por eleição e resultados numéricos centrais;
- teste automatizado do repositório;
- teste de integridade do arquivo ZIP.

A reconstrução completa a partir dos arquivos brutos do TSE está documentada, mas não foi executada integralmente no ambiente do chat. A reestimação completa a partir dos painéis processados também pode ser demorada por usar likelihood condicional exata; a validação rápida dos snapshots, figuras e PDFs foi concluída.


## Atualização v0.2.1

O manuscrito principal foi reorganizado para reduzir a fragmentação: Métodos e Resultados passaram a ter apenas três subseções cada, e a Discussão foi convertida em texto contínuo. A legenda Esquerda/Centro/Direita da Figura 6(b) foi movida para fora da área de dados.
