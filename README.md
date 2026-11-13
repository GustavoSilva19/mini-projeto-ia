
# Classificação de Dados Médicos com MLP e BBO

Mini-projeto da disciplina de Inteligência Artificial.

## Objetivo

Implementar uma rede neural Multi-Layer Perceptron (MLP)
otimizada por algoritmos Biogeography-Based Optimization
(BBO), utilizando duas bases de dados médicas.

O projeto também investiga o impacto do SMOTE no
balanceamento das classes.

## Bases de dados

- Breast Cancer Wisconsin Diagnostic (WDBC)
- Pima Indians Diabetes Dataset (PIDD)

## Algoritmos

- BBO tradicional
- ExpBBO com distribuição exponencial

## Experimentos

Cada algoritmo será avaliado:

- Sem aplicação do SMOTE.
- Com aplicação do SMOTE.

Serão utilizadas quatro configurações experimentais
para cada base de dados.

## Metodologia

- Validação cruzada estratificada de 5 partes.
- Aproximadamente 80% para treinamento e 20% para validação
em cada divisão.
- Normalização com StandardScaler.
- Imputação de valores ausentes pela mediana.
- Otimização dos pesos e vieses da MLP pelo BBO.
- Avaliação por métricas de classificação.

O SMOTE será aplicado exclusivamente aos dados de
treinamento de cada divisão.

## Métricas

- Acurácia
- Precisão
- Sensibilidade
- Especificidade
- F1-score
- G-Mean
- ROC-AUC
- MSE
- Tempo de execução

## Modificação proposta

Aplicação do SMOTE para balanceamento das classes,
comparando os resultados com a configuração original.

## Tecnologias

- Python
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn

## Como executar

1. Instale o Python.
2. Clone o repositório.
3. Crie um ambiente virtual.
4. Instale as dependências.
5. Coloque as bases de dados em data/raw.
6. Execute main.py.

### Instalação das dependências

```bash
pip install -r requirements.txt
```

### Execução

```bash
python main.py
```

## Resultados

Os resultados dos experimentos serão armazenados
na pasta results/metrics.

## Artigo científico

Li, X.-D. et al. (2022).

Multi-layer perceptron classification method of medical
data based on biogeography-based optimization algorithm
with probability distributions.

Applied Soft Computing, 121, 108766.

DOI: https://doi.org/10.1016/j.asoc.2022.108766
