# Metodologia Estatística — Benchmark de Frameworks de Federated Learning

> **Arquivo:** `validation_stats.py`  
> **Contexto:** Comparação de três frameworks de FL (Flower, NVFlare, FedBioMed) executados com **3 seeds distintos** (42, 69, 420) por configuração experimental.

---

## Sumário

1. [Visão Geral do Desenho Experimental](#1-visão-geral-do-desenho-experimental)
2. [Pré-processamento e Métricas Derivadas](#2-pré-processamento-e-métricas-derivadas)
3. [Replicabilidade Cross-Seed (CV)](#3-replicabilidade-cross-seed-cv)
4. [Estatísticas Descritivas por Configuração](#4-estatísticas-descritivas-por-configuração)
5. [Teste de Normalidade — Shapiro-Wilk](#5-teste-de-normalidade--shapiro-wilk)
6. [Comparação Global entre Frameworks](#6-comparação-global-entre-frameworks)
   - [6.1 ANOVA one-way](#61-anova-one-way)
   - [6.2 Kruskal-Wallis](#62-kruskal-wallis)
   - [6.3 Comparações pairwise](#63-comparações-pairwise)
   - [6.4 Tamanho de Efeito — Cohen's d](#64-tamanho-de-efeito--cohens-d)
   - [6.5 Correção de Múltiplos Testes — FDR Benjamini-Hochberg](#65-correção-de-múltiplos-testes--fdr-benjamini-hochberg)
7. [Comparação Matched/Blocked](#7-comparação-matchedblocked)
8. [Detecção de Outliers](#8-detecção-de-outliers)
9. [Análise de Convergência por Round](#9-análise-de-convergência-por-round)
10. [Referências](#10-referências)

---

## 1. Visão Geral do Desenho Experimental

O benchmark segue um **desenho fatorial parcial**: cada framework executa um subconjunto de configurações (combinações de `Clients × Rounds × Epochs × Batch Size`), cada uma repetida com 3 seeds aleatórias distintas. O objetivo das réplicas com seeds diferentes é:

- **Separar variabilidade determinística** (diferenças reais entre frameworks/configurações) **de variabilidade estocástica** (ruído introduzido pela inicialização aleatória dos modelos e embaralhamento dos dados).
- Permitir **inferência estatística** — sem réplicas, qualquer diferença observada pode ser mero artefato do seed.

A escolha de **3 réplicas por configuração** é um compromisso prático: é o mínimo que permite calcular variância amostral (ddof=1 exige n ≥ 2) e já oferece alguma robustez ao ruído, sendo comum em benchmarks de ML com custo computacional elevado (cada run pode durar horas).

```
Estrutura dos dados:
  └── run_metrics*.csv
        └── Framework × (Clients, Rounds, Epochs, Batch) × Seed → 1 run
```

---

## 2. Pré-processamento e Métricas Derivadas

### 2.1 Por que derivar métricas ao invés de usar colunas brutas?

Os CSVs registram métricas de infraestrutura em unidades de máquina (bytes, bps) e a acurácia como um JSON de rounds. Normalizar para unidades interpretáveis (GB, acurácia em %) e extrair pontos-chave da curva de aprendizado permite comparações diretas e coerentes.

| Métrica derivada | Origem | Justificativa |
|---|---|---|
| `final_accuracy` | último valor de `Accuracies_Per_Round` × 100 | Métrica primária de qualidade do modelo ao final do treinamento |
| `max_accuracy` | máximo de `Accuracies_Per_Round` × 100 | Captura o melhor ponto atingido, independente de oscilações nos últimos rounds |
| `first_accuracy` | primeiro valor × 100 | Representa a qualidade após um único round de comunicação |
| `convergence_delta` | `max_accuracy − first_accuracy` | Mede quanto o modelo evoluiu do round 1 ao seu pico — proxy de velocidade de convergência |
| `server_memory_gb` | `Server_Memory_Bytes / 1e9` | Unidade legível; facilita comparação com capacidade típica de servidores |
| `server_net_bps` | `Rx_Bps + Tx_Bps` | Tráfego total bidirecional — mais representativo do custo de comunicação FL que cada direção isolada |
| `duration_s` | `Adjusted_Window_s` | Janela ajustada já desconta buffers de coleta, sendo mais fiel ao tempo real de treinamento |

### 2.2 Acurácia em escala percentual

As acurácias originais estão na escala [0, 1]. O script as multiplica por 100 para ficar em [0 %, 100 %]. Isso não altera os resultados dos testes estatísticos (transformações lineares preservam ordenamento e razões), mas torna os outputs mais intuitivos para leitura humana.

---

## 3. Replicabilidade Cross-Seed (CV)

### O que é o Coeficiente de Variação (CV)?

$$CV = \frac{\sigma}{\mu}$$

O CV é a razão entre desvio padrão e média, normalizando a dispersão em relação à magnitude. Dois frameworks podem ter o mesmo desvio absoluto, mas se um opera com acurácias em torno de 90 % e outro em torno de 30 %, os impactos são muito diferentes.

### Por que usar CV ao invés do desvio padrão?

Em FL, diferentes configurações produzem acurácias médias muito diferentes (e.g., 2 clientes vs. 10 clientes). O CV permite comparar a **instabilidade relativa** entre configurações de forma justa, independente do nível absoluto de desempenho.

### Threshold de 15 %

O limiar de `CV > 0.15` (15 %) é uma convenção amplamente usada em ciências experimentais para classificar variabilidade como "alta". Configurações acima desse limiar merecem atenção: podem indicar:
- Alta sensibilidade ao seed (não-reprodutibilidade);
- Comportamento instável do framework naquela condição (e.g., convergência errática);
- Erros de execução em alguma das réplicas.

### Configurações com `n < 3`

O script sinaliza quando uma configuração tem menos de 3 seeds. Com `n = 1`, é impossível estimar variância. Com `n = 2`, a estimativa é muito ruidosa. Essa sinalização é essencial para interpretar corretamente os resultados — valores de CV e testes estatísticos para essas configurações devem ser tratados com cautela.

---

## 4. Estatísticas Descritivas por Configuração

Para cada combinação `(Framework, Clients, Rounds, Epochs, Batch)`, o script calcula:

| Estatística | Fórmula | Interpretação |
|---|---|---|
| **μ (média)** | $\bar{x} = \frac{1}{n}\sum x_i$ | Estimativa pontual do desempenho esperado |
| **σ (desvio padrão)** | $s = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}}$ | Dispersão absoluta entre seeds |
| **CV** | $s / \bar{x}$ | Dispersão relativa (ver seção 3) |
| **IC 95 %** | $\bar{x} \pm t_{0.975, n-1} \cdot \frac{s}{\sqrt{n}}$ | Intervalo onde a média verdadeira cai com 95 % de confiança |

### Por que IC t-Student ao invés de IC normal?

Com apenas 3 réplicas por configuração, `n` é pequeno demais para usar a distribuição normal (que é assintótica). A distribuição **t de Student** com `df = n − 1` graus de liberdade é a escolha correta para amostras pequenas, pois tem caudas mais pesadas — gerando intervalos mais conservadores e honestos sobre nossa incerteza.

---

## 5. Teste de Normalidade — Shapiro-Wilk

### Por que testar normalidade?

A escolha entre testes paramétricos (ANOVA, t-test) e não-paramétricos (Kruskal-Wallis, Mann-Whitney) depende de se os dados seguem uma distribuição normal. Se a normalidade não for satisfeita, os testes paramétricos perdem suas garantias teóricas de controle do erro tipo I.

### Por que Shapiro-Wilk?

O teste de **Shapiro-Wilk** é considerado o mais poderoso para detectar desvios de normalidade em **amostras pequenas** (n < 50), condição exatamente presente neste benchmark. Alternativas como Kolmogorov-Smirnov ou Anderson-Darling têm menor poder estatístico nesse regime.

$$W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

- **W próximo de 1** → dados consistentes com normalidade
- **p > 0.05** → não se rejeita a hipótese de normalidade

### Estratégia: pool por framework

O teste é aplicado sobre o **pool de todas as réplicas de um framework** (não por configuração individual, onde n = 3 seria insuficiente para detectar qualquer coisa). Isso oferece uma visão global da distribuição das métricas por framework.

> **Resultado observado:** Todas as distribuições de acurácia e duração rejeitaram normalidade (p < 0.05), justificando o uso preferencial de testes não-paramétricos na seção seguinte.

---

## 6. Comparação Global entre Frameworks

A comparação usa **dois testes complementares** para cada métrica: um paramétrico (ANOVA/t-test) e um não-paramétrico (Kruskal-Wallis/Mann-Whitney). Reportar ambos aumenta a robustez das conclusões.

### 6.1 ANOVA one-way

$$F = \frac{\text{variância entre grupos}}{\text{variância dentro dos grupos}}$$

Testa a hipótese nula de que **todas as médias são iguais** entre os frameworks. Pressupõe normalidade e homocedasticidade. É incluído para **completude e comparabilidade** com a literatura (muitos benchmarks reportam ANOVA), mas seus p-values devem ser interpretados com cautela dado que a normalidade foi rejeitada.

### 6.2 Kruskal-Wallis

É o **análogo não-paramétrico da ANOVA one-way**, baseado em ranks. Não pressupõe normalidade nem homocedasticidade.

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

- $N$ = total de observações
- $n_i$ = tamanho do grupo $i$  
- $R_i$ = soma dos ranks do grupo $i$

**Por que é o teste principal desta análise:** dados os resultados de Shapiro-Wilk (normalidade rejeitada), o Kruskal-Wallis é o teste mais adequado para detectar diferenças globais entre frameworks. Ele testa se pelo menos um framework tem distribuição estocásticamente diferente dos demais.

> **Nota de implementação:** o script trata o caso em que todos os valores são idênticos (e.g., `server_gpu_util = 0` para todos os frameworks, já que a GPU do servidor não é utilizada em FL), retornando DataFrame vazio para evitar erro do scipy.

### 6.3 Comparações pairwise

Após detectar diferença global, o passo seguinte é identificar **quais pares** de frameworks diferem.

#### Welch t-test

Variante do t-test independente que **não assume variâncias iguais** entre grupos (Satterthwaite correction). Com grupos de tamanhos e variâncias diferentes, é mais robusto que o t-test de Student padrão.

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

Incluído para compatibilidade com a literatura e como referência paramétrica.

#### Mann-Whitney U (Wilcoxon rank-sum)

Teste não-paramétrico para **duas amostras independentes**. Testa se uma distribuição é estocásticamente maior que a outra. É o complemento natural do Kruskal-Wallis para comparações pairwise quando a normalidade não é garantida.

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

**Por que reportar ambos (t-test e Mann-Whitney)?**  
Quando ambos concordam (ambos significativos ou ambos não-significativos), a conclusão é robusta. Quando divergem, sinaliza que a diferença pode ser sensível a outliers ou à forma das distribuições — informação valiosa para o pesquisador.

### 6.4 Tamanho de Efeito — Cohen's d

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}, \quad s_{pooled} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$$

**Por que reportar tamanho de efeito além do p-value?**

O p-value sozinho é insuficiente: com amostras grandes, diferenças estatisticamente significativas podem ser praticamente irrelevantes; com amostras pequenas (como aqui, n ≈ 30–50 por framework), diferenças importantes podem não atingir significância. O **Cohen's d** mede a magnitude da diferença em unidades de desvio padrão, independente do tamanho amostral.

| Valor de |d| | Classificação (Cohen, 1988) |
|---|---|
| < 0.2 | Negligível |
| 0.2 – 0.5 | Pequeno |
| 0.5 – 0.8 | Médio |
| ≥ 0.8 | Grande |

### 6.5 Correção de Múltiplos Testes — FDR Benjamini-Hochberg

Ao realizar múltiplas comparações pairwise (3 pares × N métricas), a probabilidade de pelo menos um falso positivo cresce rapidamente — problema conhecido como **inflação do erro tipo I** (ou problema do *family-wise error*).

#### Por que Benjamini-Hochberg ao invés de Bonferroni?

- **Bonferroni** controla o *family-wise error rate* (FWER): a probabilidade de **qualquer** falso positivo. É muito conservador para um grande número de testes, aumentando o erro tipo II (falsos negativos).
- **Benjamini-Hochberg (BH)** controla o *False Discovery Rate* (FDR): a **proporção esperada** de falsos positivos entre os resultados declarados significativos. É mais adequado em análises exploratórias e de benchmarking, onde algum falso positivo é tolerável, mas não queremos suprimir descobertas reais.

#### Procedimento BH

Ordenando os p-values $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$:

$$q_{(i)} = \frac{p_{(i)} \cdot m}{i}$$

Aplicando acumulação de mínimos da direita para garantir monotonicidade. Declara-se significativo todo teste com $q \leq \alpha$ (aqui, $\alpha = 0.05$).

Os **q-values** resultantes são reportados na coluna `q_fdr`, com notação de estrelas:

| Símbolo | Limiar |
|---|---|
| `***` | q < 0.001 |
| `**` | q < 0.01 |
| `*` | q < 0.05 |
| `ns` | q ≥ 0.05 |

---

## 7. Comparação Matched/Blocked

### Motivação

O benchmark **não é balanceado**: cada framework executou um subconjunto diferente de configurações. Comparar frameworks "globalmente" mistura efeitos das configurações com efeitos dos frameworks — um confundidor (confounder) clássico.

Por exemplo: se o NVFlare foi testado mais com configurações de alto desempenho (poucos clientes, muitos rounds) do que o Flower, uma diferença global de acurácia pode refletir as configurações testadas, não o framework em si.

### Estratégia: Kruskal-Wallis por Bloco

Para cada condição `(Clients, Rounds, Epochs, Batch)` onde **pelo menos dois frameworks foram executados**, aplica-se um **Kruskal-Wallis intra-bloco**. Isso controla o efeito das configurações, tornando a comparação mais justa.

Este design é análogo a um **teste de blocos aleatorizados** (Friedman test seria ainda mais rigoroso, mas requer dados completamente balanceados — condição não satisfeita aqui).

As médias e desvios padrão por framework dentro de cada bloco também são reportados para facilitar a interpretação substantiva (qual framework performou melhor naquela condição específica?).

---

## 8. Detecção de Outliers

### Dois critérios complementares

#### Regra do IQR (1.5 × IQR)

$$\text{lower} = Q_1 - 1.5 \cdot IQR \quad ; \quad \text{upper} = Q_3 + 1.5 \cdot IQR$$

Baseia-se nos quartis (mediana-centrado), sendo **resistente a outliers** na própria estimativa dos limites. É o método por trás dos whiskers do boxplot (Tukey, 1977). Adequado para distribuições assimétricas, como as de tempo de execução.

#### |Z-score| > 2.5

$$z_i = \frac{x_i - \bar{x}}{s}$$

Detecta observações a mais de 2.5 desvios padrão da média. Pressupõe distribuição aproximadamente normal; é mais sensível a outliers extremos em distribuições simétricas.

### Por que usar ambos?

Os dois métodos têm sensibilidades diferentes:

| Situação | IQR detecta? | Z-score detecta? |
|---|---|---|
| Valor extremo isolado em dist. simétrica | Sim | Sim |
| Outlier em distribuição fortemente assimétrica | Sim | Talvez não (a assimetria "absorve" o outlier na média) |
| Dois outliers do mesmo lado (mascaramento) | Pode falhar | Pode falhar |

Reportar ambos os flags (`iqr_outlier`, `z_outlier`) permite ao pesquisador avaliar a robustez da classificação — um run sinalizado por ambos os métodos merece mais atenção do que um sinalizado apenas por um.

### Aplicação por configuração, não globalmente

Os outliers são detectados **dentro de cada grupo** `(Framework, Clients, Rounds, Epochs, Batch)`. Isso é fundamental: o que seria outlier globalmente pode ser perfeitamente esperado em certas configurações (e.g., 10 clientes demorando muito mais que 2 clientes é estrutural, não anomalia).

---

## 9. Análise de Convergência por Round

### Por que analisar a curva de aprendizado e não só o resultado final?

Em FL, o **caminho até a acurácia final** é tão importante quanto o resultado final. Um framework que atinge 90 % em 3 rounds é muito mais eficiente do que outro que atinge 90 % em 7 rounds. A acurácia final sozinha não captura isso.

A análise de convergência expande os dados em formato longo (`Framework × Round → Accuracy`), permitindo:

1. **Comparar velocidade de convergência** entre frameworks com os mesmos parâmetros;
2. **Detectar instabilidade** (acurácia que sobe e desce entre rounds);
3. **Identificar o ponto de saturação** (a partir de qual round o ganho marginal é negligível).

A média e desvio padrão **por round** agregam as réplicas com seeds diferentes, mostrando a trajetória esperada e a variabilidade em cada ponto da curva.

---

## 10. Fluxo Completo da Análise

```
Dados brutos (CSVs)
       │
       ▼
 [Enrich / Parse]
 • Parse JSON de Accuracies_Per_Round
 • Extrair final_accuracy, max_accuracy, first_accuracy
 • Converter bytes → GB, calcular bps total
       │
       ▼
 [Replicabilidade]          ← CV por config × métrica
 • Flags de alta variabilidade (CV > 15%)
 • Configs com n < 3 seeds
       │
       ▼
 [Descritivas por config]   ← μ, σ, CV, IC 95% (t-Student)
       │
       ▼
 [Normalidade]              ← Shapiro-Wilk por framework
       │
       ▼
 [Comparação global]
 • ANOVA one-way  (paramétrico, referência)
 • Kruskal-Wallis (não-paramétrico, principal)    ← Normalidade rejeitada
 • Pairwise: Welch t-test + Mann-Whitney U
 • Cohen's d  (tamanho de efeito)
 • FDR Benjamini-Hochberg  (múltiplos testes)
       │
       ▼
 [Matched/Blocked]          ← Kruskal-Wallis intra-bloco
 (controla confounders de configuração)
       │
       ▼
 [Outliers]                 ← IQR 1.5× + |z| > 2.5
 (por grupo, não globalmente)
       │
       ▼
 [Convergência]             ← Acurácia por round (μ ± σ)
       │
       ▼
 Relatório + CSVs de saída
```

---

## Referências

- **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.  
  → Base para a classificação do tamanho de efeito de Cohen's d.

- **Benjamini, Y., & Hochberg, Y.** (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.  
  → Procedimento BH para correção de múltiplos testes.

- **Shapiro, S. S., & Wilk, M. B.** (1965). An Analysis of Variance Test for Normality (Complete Samples). *Biometrika*, 52(3/4), 591–611.  
  → Teste de normalidade de Shapiro-Wilk.

- **Kruskal, W. H., & Wallis, W. A.** (1952). Use of Ranks in One-Criterion Variance Analysis. *Journal of the American Statistical Association*, 47(260), 583–621.  
  → Teste de Kruskal-Wallis.

- **Mann, H. B., & Whitney, D. R.** (1947). On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other. *The Annals of Mathematical Statistics*, 18(1), 50–60.  
  → Teste Mann-Whitney U.

- **Tukey, J. W.** (1977). *Exploratory Data Analysis*. Addison-Wesley.  
  → Regra IQR para detecção de outliers.

- **Welch, B. L.** (1947). The Generalization of 'Student's' Problem when Several Different Population Variances are Involved. *Biometrika*, 34(1/2), 28–35.  
  → t-test de Welch para variâncias desiguais.
