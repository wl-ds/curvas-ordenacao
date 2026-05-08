# Avaliação de Modelos de Classificação Binária: Além da Curva ROC

Durante a formação de um cientista de dados, após a introdução aos modelos de classificação binária, a etapa seguinte consiste em aprender a avaliar o desempenho desses modelos. Em materiais didáticos como livros, cursos e publicações, são comumente apresentadas as seguintes ferramentas de avaliação:  
- Métricas:  
  - Acurácia;
  - AUC-ROC;
  - Precision (precisão);
  - Recall (sensibilidade);
  - F1-Score;
- Representações estruturais:
  - Matriz de confusão (uma tabela de contingência com Verdadeiros Positivos, Falsos Positivos, Verdadeiros Negativos e Falsos Negativos);
- Curvas de avaliação:
  - Curva ROC.  

```
Caso seja necessário relembrar essas ferramentas de avaliação, um resumo de cada uma é apresentado no glossário ao final deste material.
```
Com exceção da AUC-ROC, todas as métricas citadas, assim como a matriz de confusão, são calculadas a partir de um ponto de corte definido pelo desenvolvedor do modelo, ou seja, baseiam-se em predições já decididas. No entanto, a escolha de um ponto de corte diferente pode alterar significativamente o desempenho observado.

Nesse contexto, a curva ROC se destaca por permitir a avaliação do modelo ao longo de todos os pontos de corte possíveis. Além disso, quando há múltiplos modelos candidatos, é possível comparar seus desempenhos por meio da análise conjunta de suas curvas.

A métrica AUC-ROC, por sua vez, representa a medida agregada do desempenho de um modelo também ao longo de todos os pontos de corte possíveis e, por isso, é amplamente utilizada para a comparação direta entre modelos, uma vez que quanto maior seu valor, maior a capacidade do modelo de discriminar o desfecho.

Assim, a primeira pergunta que naturalmente surge é: para escolher o melhor modelo, basta calcular a AUC-ROC e selecionar aquele com o maior valor? A resposta é: não.

Como a AUC-ROC é uma medida agregada, um valor mais elevado não garante, necessariamente, melhor desempenho em todas as situações. Isso ocorre porque as curvas ROC de diferentes modelos podem se cruzar, indicando que cada modelo pode ser superior em determinados intervalos de pontos de corte. Daí a importância da análise das curvas: é necessário avaliar qual modelo apresenta melhor desempenho nos pontos de corte que são, de fato, relevantes para a área de negócio.

Apesar de ser uma excelente ferramenta de avaliação, a curva ROC é uma linguagem mais voltada ao cientista de dados. Ao final do desenvolvimento de um modelo, toda decisão deverá estar alinhada à sua aplicação para que sejam alcançados os objetivos do negócio. Imagine que um gestor de crédito faça a seguinte pergunta sobre um modelo: "Se eu rejeitar os 20% mais arriscados da carteira, quantos maus pagadores eu evito?".

Responder a essa pergunta utilizando apenas a curva ROC não é uma tarefa trivial. Embora a curva permita avaliar o desempenho do modelo ao longo de diferentes pontos de corte, ela não responde claramente quantos maus pagadores estão concentrados nos segmentos mais arriscados da base.

Do ponto de vista do negócio, é necessário ir além da curva ROC, pois o interesse não está apenas em avaliar a capacidade geral de discriminação do modelo, mas também em entender como os indivíduos se distribuem quando ordenados pelo risco predito. É nesse cenário que as curvas de ordenação se tornam fundamentais, pois traduzem o desempenho do modelo em termos diretamente aplicáveis à tomada de decisão.

Ao ordenar os indivíduos de acordo com o risco predito pelo modelo, é possível analisar o comportamento acumulado dos eventos e não eventos ao longo da amostra, o que permite construir diferentes ferramentas de avaliação.

## Da discriminação à ordenação: ampliando a caixa de ferramentas de avaliação
A avaliação de classificadores binários a partir da abordagem de ordenação é amplamente utilizada em contextos de negócio, especialmente em aplicações de crédito e marketing. Ainda que seja predominante nesses contextos, essa abordagem é pouco encontrada em materiais técnicos. Nesta seção, essas ferramentas serão apresentadas sob uma perspectiva técnica, formalizada de maneira rigorosa, evidenciando como a ordenação das observações constitui a base para a construção de diferentes curvas de avaliação.

As duas métricas mais utilizadas na avaliação de modelos de crédito é a estatística de **Kolmogorov–Smirnov (KS)** e **Gini**. Apesar de serem métricas de discriminação, elas são derivadas da ordenação, uma vez que são obtidas após ordenarmos as probabilidades.

O KS é uma métrica calculada a partir das mesmas taxas acumuladas utilizadas na construção da curva ROC: taxa de verdadeiros positivos (*true positive rate* - TPR) e a taxa de falsos positivos (*false positive rate* - FPR). Enquanto a curva ROC descreve o desempenho do modelo por meio da relação entre a TPR e a FPR ao longo de diferentes pontos de corte, a métrica KS pode ser interpretada como o módulo da maior distância entre as curvas acumuladas da TPR e FPR quando as observações são ordenadas segundo a probabilidade predita, estabelecendo uma conexão direta entre ambas.

O Gini é uma transformação linear da AUC-ROC, obtida pela equação **Gini = 2 × AUC − 1**. Por ser apenas uma mudança de escala, as duas métricas carregam exatamente a mesma informação sobre o poder discriminante do modelo, ou seja, qualquer ordenação de modelos feito com AUC produz o mesmo resultado com Gini, sem exceção. A vantagem do Gini está na interpretação: na AUC, 0,5 é o ponto de referência do acaso, o que pode parecer contraintuitivo, enquanto no Gini esse ponto é o zero, tornando a escala mais intuitiva. Assim, modelos melhores que o acaso assumem valores entre 0 e 1, e modelos com ordenação invertida assumem valores negativos, facilitando a comunicação com a área de negócio.

Já as ferramentas de visualização mais comuns são as curvas de **Ordenação Relativa** (Taxa de Maus por Percentil), **Ganho acumulado** (também chamada de *Cumulative Gain*, Perfil de Eficiência Acumulada ou *Cumulative Accuracy Profile* - CAP), ***Lift***, e **Curva de Inadimplência Acumulada** (*Cumulative Bad Rate Curve*). Essas curvas analisam o desempenho do modelo após ordenar as observações pela probabilidade predita, avaliando a concentração dos eventos de interesse ao longo da ordenação produzida pelo modelo. Essa análise pode ser realizada diretamente sobre as observações ranqueadas ou por meio de agrupamentos, dados por percentis ou decis.

A partir desse conjunto de ferramentas, pode-se avaliar distintos aspectos de desempenho, como separação entre distribuições, concentração de eventos, coerência de ordenação e comportamento para diferentes limiares de decisão. Todas refletem a capacidade do modelo de classificar corretamente eventos e não eventos.

Antes de detalharmos essas curvas e apresentarmos exemplos, é necessário formalizar como elas são construídas. Na tabela a seguir, utilizamos um conjunto de 10 casos para ilustrar, de forma prática, as relações envolvidas e o procedimento pelo qual essas avaliações visuais são obtidas.

#### Tabela 1. Exemplo de resultado com 10 observações.
| i | $y_i$ | $p_i$ | $1-p_i$ | $\hat{y}_i$ | $Pop_i^{cum}$ | $TP_i$ | $FP_i$ | $TPR_i$ | $FPR_i$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 95% | 5% | 1 | 10.0% | 1 | 0 | 20.0% | 0.0% |
| 2 | 1 | 80% | 20% | 1 | 20.0% | 2 | 0 | 40.0% | 0.0% |
| 3 | 0 | 75% | 25% | 1 | 30.0% | 2 | 1 | 40.0% | 20.0% |
| 4 | 1 | 60% | 40% | 1 | 40.0% | 3 | 1 | 60.0% | 20.0% |
| 5 | 1 | 50% | 50% | 1 | 50.0% | 4 | 1 | 80.0% | 20.0% |
| 6 | 0 | 45% | 55% | 0 | 60.0% | 4 | 2 | 80.0% | 40.0% |
| 7 | 0 | 30% | 70% | 0 | 70.0% | 4 | 3 | 80.0% | 60.0% |
| 8 | 1 | 25% | 75% | 0 | 80.0% | 5 | 3 | 100.0% | 60.0% |
| 9 | 0 | 20% | 80% | 0 | 90.0% | 5 | 4 | 100.0% | 80.0% |
| 10 | 0 | 10% | 90% | 0 | 100.0% | 5 | 5 | 100.0% | 100.0% |
> **Em que:**\
> $y_i$ = valor real (0 ou 1);\
> $p_i$ = probabilidade predita pelo modelo;\
> $\hat{y}_i$ = classe predita pelo modelo;\
> $Pop_i^{cum}$ = percentual acumulado da população;\
> $TP_i$ = verdadeiros positivos acumulados;\
> $FP_i$ = falsos positivos acumulados;\
> $TPR_i$ = taxa de verdadeiros positivos (sensibilidade);\
> $FPR_i$ = taxa de falsos positivos (1 - especificidade).

Ao ordenar os casos segundo a probabilidade predita

$$
p_i = P(y=1|x_i)
$$

temos a ordenação das observações, o que permite avaliar o comportamento acumulado dos eventos e não eventos ao longo da amostra. Assim, definimos:

- $n$: total de observações  
- $n_1$: total de eventos  
- $n_0$: total de não eventos  

Para cada posição ordenada, temos as taxas acumuladas de eventos e não eventos:

$$
TPR_i = \frac{TP_i}{n_1}
$$

$$
FPR_i = \frac{FP_i}{n_0}
$$

em que:

- $TP_i$ = número acumulado de eventos até a posição $i$  
- $FP_i$ = número acumulado de não eventos até a posição $i$

A partir dessas definições é possível construir a curva ROC, que representa graficamente todos os pares $(FPR(t), TPR(t))$ obtidos para diferentes limiares de decisão $t$ aplicados às probabilidades estimadas.

Seguindo essa mesma lógica, obtém-se o gráfico de Kolmogorov–Smirnov (KS), que mede a maior distância entre as distribuições acumuladas de eventos ($TPR$) e não eventos ($FPR$):

A tabela anterior descreve o comportamento do modelo no nível das observações $i$, ordenadas de forma decrescente segundo a probabilidade estimada $p_i$. Essa representação permite calcular diretamente métricas como as taxas acumuladas $TPR_i$, $FPR_i$ e o *Lift*. Para facilitar a interpretação e a construção de gráficos como o de ordenação, é comum agrupar as observações em faixas, geralmente percentis ou decis. Desse modo, seja $g = 1, \dots, G$ o índice desses agrupamentos, definidos segundo a ordenação das probabilidades $p_i$, temos cada grupo $g$ enquanto um subconjunto de observações da tabela original.

**Tabela 2.** Resultado do modelo agrupado em quintil.
| $g$ | $n_g$ | $e_g$ | $ne_g$ | $Pop_g$ | $Pop_g^{cum}$ | $TP_g$ | $FP_g$ | $TPR_g$ | $FPR_g$ | $Lift_g$ | $err_g^{cum}$ | $er_g$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | 0 | 0.2 | 0.2 | 2 | 0 | 0.40 | 0.00 | 2.00 | 0.500 | 1.00 |
| 2 | 2 | 1 | 1 | 0.2 | 0.4 | 3 | 1 | 0.60 | 0.20 | 1.50 | 0.375 | 0.50 |
| 3 | 2 | 1 | 1 | 0.2 | 0.6 | 4 | 2 | 0.80 | 0.40 | 1.33 | 0.333 | 0.50 |
| 4 | 2 | 1 | 1 | 0.2 | 0.8 | 5 | 3 | 1.00 | 0.60 | 1.25 | 0.250 | 0.50 |
| 5 | 2 | 0 | 2 | 0.2 | 1.0 | 5 | 5 | 1.00 | 1.00 | 1.00 | 0.000 | 0.00 |
> **Em que:**\
> $g$ = índice do grupo (quintil);\
> $n_g$ = número de observações no grupo;\
> $e_g$ = número de eventos (y=1) no grupo;\
> $ne_g$ = número de não eventos (y=0) no grupo;\
> $Pop_g$ = proporção da população no grupo;\
> $Pop_g^{cum}$ = proporção acumulada da população;\
> $TP_g$ = verdadeiros positivos acumulados;\
> $FP_g$ = falsos positivos acumulados;\
> $TPR_g$ = taxa de verdadeiros positivos (sensibilidade);\
> $FPR_g$ = taxa de falsos positivos (1 - especificidade);\
> $Lift_g$ = *lift* do grupo;\
> $err_g^{cum}$ = taxa de evento acumulada;\
> $er_g$ = taxa de evento no grupo.

A agregação apresenta essas estatísticas indexadas em $g$. Nela, $n_g$ representa o número de observações dentro da faixas, enquanto que $e_g$ e $ne_g$ representam, respectivamente, o número de eventos e não eventos observados em cada faixa. A partir dessas quantidades são definidos os acumulados:

$$
e_g^{cum} = \sum_{k=1}^{g} e_k = TP_g
$$

$$
ne_g^{cum} = \sum_{k=1}^{g} ne_k = FP_g
$$

Com base nesses acumulados, podem ser obtidas as taxas acumuladas de eventos e não eventos, de forma bastante similar a tabela original:

$$
TPR_g = \frac{TP_g}{n_1}
$$

$$
FPR_g = \frac{FP_g}{n_0}
$$

A tabela agregada também permite calcular o *Lift* por faixa, definido de forma simplidicada como a razão entre a taxa acumulada de eventos e a fração acumulada da população:

$$
Lift_g =
\frac{TPR_g}{Pop_g^{cum}}
$$

Além disso, são apresentadas medidas adicionais utilizadas em análises de risco de crédito, como a taxa de eventos por faixa, chamada de curva de ordenação

$$
er_g = \frac{e_g}{n_g}
$$

e a taxa acumulada reversa de eventos ou eventos acumuladas até o evento médio da amostra dado por

$$
err_g^{cum} = \frac{\sum_{j \geq g} e_j}{\sum_{j \geq g} n_j}
$$

Do ponto de vista conceitual, a tabela agregada resume a informação contida na tabela original e facilita a criação das visualizações das curvas. Em termos matemáticos, ela pode ser entendida como uma agregação do ranqueamento definido pelas probabilidades $p_i$, preservando as quantidades acumuladas que dão origem às informações avaliadas.

## Ferramentas visuais de ordenação
  
Partindo das discussões realizadas até aqui, utilizamos o *dataset* público *Give Me Some Credit** para trabalharmos a partir da saída real de um modelo. Trata-se de um problema clássico de risco de crédito, composto principalmente por variáveis relacionadas ao comportamento financeiro dos indivíduos.

```
*Dataset disponível em https://www.kaggle.com/c/GiveMeSomeCredit
```
  
A variável resposta (*target*) é uma *dummy* que classifica os clientes em bons e maus pagadores, sendo considerados maus aqueles que apresentaram atrasos superiores a 90 dias no pagamento de suas obrigações financeiras. Por se tratar de um problema de classificação binária, foram treinados dois modelos amplamente utilizados no mercado: a Regressão Logística e o Random Forest Classifier.

Cabe destacar que, em modelos de crédito, existe a convenção de que clientes mais arriscados recebem menores *scores*, enquanto os menos arriscados recebem maiores *scores*. Portanto, a probabilidade predita utilizada nos gráficos refere-se à probabilidade de o cliente ser bom, de forma que o decil 1 representa os clientes com maior probabilidade de serem maus, e o decil 10, aqueles com maior probabilidade de serem bons.

A seguir, começamos a apresentar as curvas de ordenação utilizando os resultados dos dois modelos de exemplo. Além de nossa análise, também fizemos a aplicação de LLM para a interpretação dos gráficos para demonstrar o potencial do uso dessa ferramenta.

### Curva de Ordenação Relativa

O gráfico de ordenação relativa apresenta a taxa de inadimplência (*bad rate*) ao longo dos decis, definidos a partir da ordenação dos clientes pela probabilidade predita dos modelos. Em um cenário ideal, espera-se uma curva monotônica, na qual os primeiros decis concentrem os indivíduos de maior risco, enquanto os decis finais concentrem aqueles de menor risco, evidenciando boa capacidade de discriminação.

Nos modelos avaliados, ambos capturam o comportamento esperado de ordenação, diferenciando clientes bons e maus ao longo dos decis. No entanto, há diferenças relevantes na qualidade e na consistência dessa separação.
<div align="center">
<img width="450" height="369" alt="Image" src="https://github.com/user-attachments/assets/f531077d-02d3-49b0-97ab-85d5e27d6024" />

  <p><strong>Figura 1.</strong> Curva de ordenação relativa.</p>
</div>

A Regressão Logística apresenta uma curva mais suave, com taxa de inadimplência de aproximadamente 26% no primeiro decil — cerca de 3,9 vezes a média da carteira (6,68%). Apesar de manter a tendência decrescente ao longo da maior parte da distribuição, o modelo exibe perda de ordenação nos decis finais, evidenciada pela inversão da taxa de inadimplência no último decil, que sobe para aproximadamente 5% após atingir o mínimo no nono decil (~2,5%). Essa inversão, ainda que pequena, compromete a monotonicidade da curva e pode limitar o uso do score em decisões que dependem da confiabilidade do ranqueamento nos melhores perfis.

A Random Forest demonstra poder de discriminação superior, concentrando aproximadamente 37% de maus pagadores no primeiro decil — cerca de 5,5 vezes a média e 42% acima da taxa observada pela Regressão Logística no mesmo decil. A partir do quarto decil, sua taxa de inadimplência se mantém abaixo da média geral e consistentemente inferior à da Regressão Logística, refletindo também uma melhor identificação dos clientes de baixo risco. A ausência de inversões ao longo da curva reforça a robustez da ordenação produzida pelo modelo.

De forma geral, a Random Forest apresenta desempenho superior em ordenação e discriminação, com maior concentração de inadimplentes nos piores decis, menor taxa de inadimplência nos melhores decis e monotonicidade preservada em toda a curva. Esses atributos a tornam mais confiável para segmentação e definição de políticas de crédito, enquanto a Regressão Logística, embora apresente ordenação razoável, mostra-se menos eficaz tanto na captura dos perfis de maior risco quanto na consistência do ranqueamento nos melhores decis.

Ao utilizarmos um prompt padrão para interpretação do gráfico a partir da LLM, obtivemos a seguinte resposta:
 
>Ambos os modelos apresentam ordenação adequada, com maior *bad rate* nos percentis iniciais e redução consistente ao longo das faixas. O Random Forest exibe maior separação entre os extremos, com *bad rate* mais alto no primeiro percentil e mais baixo nos percentis finais em comparação à Regressão Logística. Portanto, com base na métrica apresentada, o Random Forest demonstra superioridade na discriminação de risco.


### Curva de Ganho Acumulado

A Curva de Ganho Acumulado (também conhecida como *CAP Curve* em risco de crédito) apresenta a proporção acumulada de maus pagadores capturados conforme a população é ordenada do pior para o melhor score predito pelo modelo.

Espera-se que modelos bem ajustados apresentem curvas mais côncavas, com crescimento acentuado nos primeiros decis, indicando maior capacidade de capturar os indivíduos de maior risco logo nas primeiras posições do ranqueamento. Quanto mais próxima a curva estiver do canto superior esquerdo, maior o poder discriminatório do modelo. É importante observar que, supondo uma diagonal de referência (que representaria uma seleção aleatória), a área entre a curva do modelo e essa diagonal está diretamente relacionada ao **Gini**, citada anteriormente como uma das principais métricas-resumo de discriminação.

<div align="center">
<img width="457" height="369" alt="Image" src="https://github.com/user-attachments/assets/01d4416e-348e-4c02-a834-9160ead6f698" />

  <p><strong>Figura 2.</strong> Curva de ganho acumulado.</p>
</div>

A Random Forest demonstra desempenho superior ao longo de toda a curva, com destaque para os decis iniciais. Aproximadamente 55% dos maus pagadores são capturados nos primeiros 10% da população, percentual que ultrapassa 70% ao considerar os 20% piores scores. Esse comportamento evidencia forte capacidade de priorização dos indivíduos de maior risco e se traduz em uma área sob a curva significativamente maior, refletindo um Gini mais elevado.

A Regressão Logística, por sua vez, apresenta uma curva com crescimento menos acentuado, capturando aproximadamente 40% dos maus pagadores nos primeiros 10% e exigindo uma parcela maior da população para atingir níveis de captura semelhantes aos do Random Forest. Essa diferença na concavidade das curvas indica menor poder discriminatório global, o que se reflete em um Gini inferior.

Ao utilizarmos um prompt padrão para interpretação do gráfico a partir da LLM, obtivemos a seguinte resposta:

>O gráfico exibe a métrica de gain acumulado ao longo dos percentis de score, mostrando que ambos os modelos apresentam crescimento monotônico, indicando ordenação consistente do risco. O modelo Random Forest apresenta maior separação entre as faixas de maior e menor risco, com ganho mais acentuado nos percentis iniciais em comparação à Regressão Logística. Portanto, com base na métrica apresentada, o Random Forest demonstra desempenho superior em discriminar risco ao longo dos percentis.


### Curva de *Lift*

A Curva de *Lift* (ou *Cumulative Lift Chart*) apresenta o ganho do modelo ao longo dos decis de score em comparação a uma seleção aleatória. O *lift* é calculado como a razão entre a proporção acumulada de maus capturados e a proporção acumulada da população, indicando quantas vezes o modelo é mais eficiente do que uma abordagem aleatória.

Por construção, a curva parte do valor máximo no primeiro decil e converge para 1 no último decil, ponto em que toda a população foi considerada e o ganho sobre o aleatório desaparece.

<div align="center">
<img width="443" height="369" alt="Image" src="https://github.com/user-attachments/assets/b96bc43d-c4d5-4262-8226-917fa043082b" />

  <p><strong>Figura 3.</strong> Curva de <em>lift.</em></p>
</div>

O modelo Random Forest apresenta desempenho superior ao longo de toda a ordenação, com uma curva consistentemente acima da Regressão Logística. Nos decis iniciais, o modelo atinge níveis de *lift* significativamente mais elevados, indicando maior concentração de maus pagadores em relação à proporção da população analisada.

No primeiro decil, o Random Forest identifica aproximadamente 5,5 vezes mais maus pagadores do que uma seleção aleatória, enquanto a Regressão Logística atinge cerca de 3,9 vezes, ou seja, o *lift* é cerca de 40% superior em favor do modelo Random Forest. Essa vantagem se mantém ao longo dos decis intermediários e só se dissipa nos últimos decis, onde ambas as curvas convergem naturalmente para 1. O resultado reforça a maior eficiência do Random Forest na priorização de clientes de alto risco.

Ao utilizarmos um prompt padrão para interpretação do gráfico a partir da LLM, obtivemos a seguinte resposta:

>O gráfico mostra que ambos os modelos apresentam ordenação consistente de risco, com o *lift* decrescendo monotonicamente ao longo dos percentis. O Random Forest exibe maior *lift* nas faixas iniciais (maior risco) e mantém superioridade em todos os percentis em relação à Regressão Logística. Portanto, o Random Forest demonstra melhor separação de risco entre as faixas, sendo superior segundo a métrica apresentada.


### Curva de Inadimplência Acumulada 
A Curva de Inadimplência Acumulada (ou *Cumulative Bad Rate Chart*) apresenta a taxa de inadimplência observada na carteira acumulada à medida que a população é incorporada dos melhores para os piores scores (ao contrário dos gráficos anteriores, aqui vemos no eixo x a ordem inversa de decis: 10 → 1). Cada ponto da curva responde à pergunta: "Se eu aprovar os X% melhores scores, qual seria a taxa de inadimplência da carteira resultante?". Essa visualização é particularmente útil para apoiar decisões de política de crédito, pois permite avaliar o trade-off entre volume de aprovação e qualidade da carteira.

<div align="center">
<img width="443" height="369" alt="Image" src="https://github.com/user-attachments/assets/a299c983-3b25-499c-bc1c-ef7893b77fab" />

  <p><strong>Figura 4.</strong> Curva de inadimplência acumulada.</p>
</div>

O Random Forest mantém a taxa de inadimplência acumulada consistentemente inferior à da Regressão Logística ao longo de toda a curva. No ponto de corte destacado (decil 5), que corresponde à aprovação dos 50% melhores scores, o Random Forest apresenta inadimplência de 1,30% contra 3,48% da Regressão Logística, uma diferença absoluta de 2,18 p.p. e uma redução relativa de aproximadamente 63% no risco da carteira aprovada. Ambas as inadimplências estão consideravelmente abaixo da média geral da carteira (6,68%), mas a vantagem do Random Forest é expressiva.

Em termos práticos, o Random Forest permite avançar por mais decis mantendo uma inadimplência relativamente muito inferior à Regressão Logística, ampliando o espaço de decisão para a área de política de crédito. Por exemplo, é possível buscar um ponto de corte que maximize o volume de aprovação sujeito a uma inadimplência alvo, ou então reduzir o risco da carteira para um mesmo volume de aprovação. Essa flexibilidade reforça a superioridade do modelo baseado em árvores na identificação e priorização de indivíduos com melhor perfil de risco.

Ao utilizarmos um prompt padrão para interpretação do gráfico a partir da LLM, obtivemos a seguinte resposta:

>O modelo Random Forest apresenta maior separação de risco entre os percentis, com taxas de maus (*bad rate*) significativamente menores nas faixas de menor risco e maiores nas faixas de maior risco, em comparação à Regressão Logística. Observa-se que a curva do Random Forest mantém uma tendência crescente mais acentuada e consistente ao longo dos percentis, enquanto a Regressão Logística apresenta menor variação entre as faixas. Portanto, com base na métrica de *bad rate*, o Random Forest demonstra ordenação e discriminação superiores entre os grupos de risco.


## Conclusão
Em síntese, a avaliação de modelos de classificação binária não deve ser vista como uma escolha entre rigor técnico e aplicabilidade prática, mas como uma combinação dessas perspectivas. Enquanto métricas como AUC-ROC, KS e Gini sintetizam a capacidade do modelo de ordenar corretamente eventos e não eventos, a abordagem baseada na ordenação das probabilidades preditas torna essa capacidade observável ao longo da amostra, permitindo analisar como o risco se concentra e evolui no ranqueamento. Ferramentas como as curvas de ordenação relativa, ganho acumulado, *lift* e inadimplência acumulada permitem avaliar, de forma objetiva, o impacto de diferentes estratégias de corte sobre o negócio. Dessa forma, a ordenação não substitui a análise de discriminação, mas a operacionaliza, preservando o rigor estatístico e, assim, aproximando a modelagem e os objetivos estratégicos.

# GLOSSÁRIO

**Acurácia**: número de predições corretas (Verdadeiros Positivos e Verdadeiros Negativos) dividido pelo número de todas as observações (todas as entradas da matriz de confusão somadas). É dada pela equação: $$\frac{(VP + VN)}{(VP + VN + FP + FN)}$$


***Precision* (precisão)**: mede quantas das observações preditas como positivas são de fato positivas. É muito usada para limitar o número de falsos positivos. Também conhecida como VPP (valor preditivo positivo), é dada pela equação: $$\frac{VP}{(VP + FP)}$$
  

***Recall* (sensibilidade)**: mede a capacidade de um modelo identificar corretamente os casos positivos, ou seja, quantas das observações positivas foram capturadas pelas predição positivas. É usada quando há necessidade de identificar a maior parte das observações positivas (por positivo, queremos dizer desfecho de interesse, ou o que queremos predizer). Também conhecida como TPR (*True Positive Rate* ou Taxa de Verdadeiros Positivos - VPP), é dada pela equação: $$\frac{VP}{(VP + FN)}$$

**F<sub>1</sub>-score**: métrica que combina *precision* e *recall*. Corresponde à média harmônica entre essas duas medidas:

$$F_1 = 2 \cdot \frac{(Precision \cdot Recall)}{(Precision + Recall)}$$

Também existem variações como o $F_{0,5}-score$ e o $F_2-score$. A diferença entre elas está no parâmetro *β*, já que o F-score é definido, de forma geral, por:  

$$ F_\beta = \frac{(1 + \beta^2) \cdot (Precision \cdot Recall)}{(\beta^2 \cdot Precision) + Recall} $$

De modo geral, $\beta < 1$ atribui maior peso ao *precision*, enquanto $\beta > 1$ enfatiza o *recall*. Na literatura, é comum encontrar a afirmação de que essa métrica é especialmente útil em cenários com dados desbalanceados. Não aprofundamos essa discussão por não ser o foco deste projeto; ainda assim, mostramos, por meio de uma abordagem baseada em ordenação, que é possível lidar com desbalanceamento mesmo sem recorrer a essa métrica.
