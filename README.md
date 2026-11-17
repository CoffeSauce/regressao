# Desenvolvimento de Modelo Preditivo Baseado em Regressão para Análise de Dados Tabulares (California Housing)

Projeto desenvolvido para a **Unidade III da disciplina de Aprendizado Profundo (2026.2)**, na categoria **Regressão e Dados Tabulares**.

---

## 1. Introdução

A precificação de imóveis é um problema clássico de regressão em dados tabulares que envolve interações complexas e não lineares entre fatores socioeconômicos, estruturais e geográficos. Enquanto modelos lineares tradicionais assumem uma relação puramente aditiva e proporcional entre as variáveis e o preço, redes neurais profundas conseguem aprender representações intermediárias capazes de capturar efeitos combinados — como a valorização não linear associada à renda média de um bairro em conjunto com sua localização geográfica.

Neste trabalho, desenvolveu-se uma aplicação computacional baseada em **Redes Neurais Artificiais (Multilayer Perceptron — MLP)** para estimar o valor mediano das residências (`MedHouseVal`) em diferentes setores censitários (*block groups*) da Califórnia. Para validar a eficácia da abordagem de Aprendizado Profundo, o desempenho da rede neural foi comparado diretamente com um modelo clássico de **Regressão Linear** (*baseline*).

---

## 2. Desenvolvimento e Técnicas Utilizadas

### 2.1. Conjunto de Dados e Análise Exploratória
Utilizou-se o dataset **California Housing** (Pace & Barry, 1997), obtido via `scikit-learn`, composto por **20.640 registros** e **9 atributos** (8 variáveis preditoras numéricas e 1 variável alvo), sem valores ausentes:

| Variável | Descrição |
| :--- | :--- |
| `MedInc` | Renda mediana dos moradores do setor (em dezenas de milhares de US$) |
| `HouseAge` | Idade mediana das residências no setor (em anos) |
| `AveRooms` | Número médio de cômodos por domicílio |
| `AveBedrms` | Número médio de quartos por domicílio |
| `Population` | População total residente no setor |
| `AveOccup` | Número médio de ocupantes por domicílio |
| `Latitude` | Latitude geográfica do setor |
| `Longitude` | Longitude geográfica do setor |
| **`MedHouseVal`** | **Variável alvo: valor mediano das casas (em centenas de milhares de US$)** |

A análise de correlação de Pearson mostrou que a **renda mediana (`MedInc`, $r \approx 0,69$)** é o atributo linearmente mais associado ao valor dos imóveis, seguida pelo número médio de cômodos (`AveRooms`) e pela localização geográfica (`Latitude` e `Longitude`).

### 2.2. Pré-processamento e Divisão dos Dados
* **Particionamento:** Os dados foram divididos em **80% para treino** (16.512 amostras) e **20% para teste** (4.128 amostras), utilizando semente fixa (`random_state=42`) para reprodutibilidade. Durante o treinamento da rede neural, 20% do conjunto de treino foi reservado para validação (`validation_split=0.2`).
* **Padronização (Z-score):** Como as variáveis possuem escalas muito distintas (por exemplo, `Population` na casa dos milhares e `AveBedrms` próximo de 1), aplicou-se o `StandardScaler` ($\mu = 0, \sigma = 1$). O ajuste (`fit`) foi realizado **exclusivamente sobre o conjunto de treino** e posteriormente aplicado ao conjunto de teste (`transform`), evitando vazamento de dados (*data leakage*).

### 2.3. Arquitetura da Rede Neural (MLP) e Treinamento
Implementou-se uma rede neural *feedforward* (MLP) em **TensorFlow / Keras** com **2.689 parâmetros treináveis**:

1. **Camada de Entrada:** 8 neurônios correspondentes às variáveis padronizadas.
2. **Camada Oculta 1:** Densa com **64 neurônios** e função de ativação **ReLU** (576 parâmetros).
3. **Camada Oculta 2:** Densa com **32 neurônios** e função de ativação **ReLU** (2.080 parâmetros).
4. **Camada de Saída:** Densa com **1 neurônio** sem função de ativação (saída linear contínua para regressão, 33 parâmetros).

**Hiperparâmetros e Regularização:**
* **Função de Perda:** Erro Quadrático Médio (`MSE`).
* **Otimizador:** `Adam` com tamanho de lote (`batch_size`) igual a `32`.
* **Controle de Overfitting:** Utilizou-se `EarlyStopping` monitorando `val_loss` com paciência de 10 épocas e restauração automática dos melhores pesos (`restore_best_weights=True`). O treinamento atingiu o menor erro de validação na **Época 34** (`val_loss = 0,3003`, `val_mae = 0,3818`) e foi encerrado automaticamente na **Época 44**.

---

## 3. Resultados, Análise Crítica e Conclusões

### 3.1. Métricas de Desempenho no Conjunto de Teste (4.128 amostras)
Ambos os modelos foram avaliados sobre dados inéditos utilizando o Erro Médio Absoluto (**MAE**), a Raiz do Erro Quadrático Médio (**RMSE**) e o Coeficiente de Determinação (**$R^2$**):

| Modelo | MAE (US$ 100k) | Erro Médio (US$) | RMSE (US$ 100k) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| Regressão Linear (*Baseline*) | 0,533 | ~US$ 53.300 | 0,746 | 0,576 |
| **Rede Neural (MLP)** | **0,374** | **~US$ 37.400** | **0,538** | **0,779** |

### 3.2. Demonstração Prática (Inferência em Amostra Inédita)
Ao submeter um setor censitário inédito do conjunto de teste (índice `20046`: `MedInc = 1,68`, `HouseAge = 25`, `AveRooms = 4,19`, `Latitude = 36,06`, `Longitude = -119,01`) ao pipeline completo (`StandardScaler` + MLP):
* **Valor Real:** US$ 47.700
* **Valor Previsto pela Rede Neural:** US$ 39.654 (diferença absoluta de US$ 8.046)

### 3.3. Análise Crítica
1. **Superioridade da Abordagem Não Linear:** A Rede Neural reduziu o **MAE em 29,8%** e o **RMSE em 27,9%** em relação à Regressão Linear, elevando o poder de explicação da variância ($R^2$) de **57,6% para 77,9%**. Isso confirma que a combinação não linear entre renda e coordenadas geográficas é determinante para a precificação imobiliária.
2. **Estabilidade de Convergência:** A curva de aprendizado demonstrou rápida redução do erro nas primeiras 20 épocas seguida de estabilização entre as épocas 25 e 44, sem divergência entre treino e validação, atestando a eficácia da normalização e do `EarlyStopping`.
3. **Limitação Estrutural do Dataset (Efeito Teto):** No gráfico de valores reais versus preditos, observa-se uma linha vertical acentuada em `y = 5,0` (US$ 500.000). Isso ocorre porque o censo original aplicou um truncamento (*capping*) nos imóveis acima de US$ 500 mil, limitando a capacidade de qualquer modelo estimar com precisão propriedades de alto padrão acima desse patamar.

### 3.4. Conclusões e Trabalhos Futuros
O modelo baseado em Aprendizado Profundo atingiu o objetivo proposto, superando significativamente o *baseline* linear em todas as métricas avaliadas. Como trabalhos futuros, sugere-se a criação de atributos espaciais explícitos (distância aos principais centros econômicos, como São Francisco e Los Angeles), o teste de arquiteturas tabulares com conexões residuais (*Tabular ResNet*) e o uso de funções de perda robustas a *outliers* (como *Huber Loss*).

---

## 4. Tecnologias e Referências
* **Linguagem e Bibliotecas:** Python 3, NumPy, Pandas, Matplotlib, Scikit-Learn e TensorFlow / Keras.
* **Dataset:** Pace, R. Kelley, and Ronald Barry. *"Sparse spatial autoregressions."* Statistics & Probability Letters 33.3 (1997): 291-297.
