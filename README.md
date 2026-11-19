# Mini-Projeto de Inteligência Artificial

## Classificação de Dados Médicos utilizando MLP e Otimização por Biogeografia (BBO)

Este projeto foi desenvolvido para a disciplina de Inteligência Artificial do curso de Ciência da Computação da Universidade Católica de Pernambuco (UNICAP).

O objetivo é investigar a utilização do algoritmo de Otimização Baseada em Biogeografia (Biogeography-Based Optimization — BBO) no treinamento de uma rede neural Multi-Layer Perceptron (MLP), aplicada à classificação de dados médicos.

O trabalho consiste em uma adaptação simplificada da metodologia apresentada no artigo científico de referência, incluindo uma análise adicional do impacto da técnica SMOTE no balanceamento dos dados.

## Integrantes

Projeto desenvolvido pelos alunos do curso de Ciência da Computação da Universidade Católica de Pernambuco (UNICAP), para a disciplina de Inteligência Artificial.

- Camila Torquato

- Gustavo Igor

- Rielly Luiza

- Samuel Ponciano

**Instituição:** Universidade Católica de Pernambuco (UNICAP)

**Curso:** Ciência da Computação

**Disciplina:** Inteligência Artificial

**Docente:** Francisco Madeiro

**Ano:** 2026

## Artigo de referência

**Título:** Multi-layer perceptron classification method of medical data based on biogeography-based optimization algorithm with probability distributions

**Autores:** Xu-Dong Li, Jie-Sheng Wang, Wen-Kuo Hao, Min Wang e Min Zhang.

**Periódico:** Applied Soft Computing, v. 121, art. 108766, 2022.

**DOI:** https://doi.org/10.1016/j.asoc.2022.108766

O artigo investiga variantes do algoritmo BBO com distribuições de probabilidade para otimizar redes MLP na classificação de dados médicos.

Neste mini-projeto, foram implementadas duas configurações: BBO padrão e BBO com distribuição exponencial.

## Tecnologias utilizadas

- Python 3.13
- NumPy e Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib

## Bases de dados

Foram utilizadas duas bases de dados:

**Breast Cancer Wisconsin Diagnostic (WDBC):** contém 569 registros, 30 atributos preditivos e duas classes relacionadas ao diagnóstico de tumores (357 benignos e 212 malignos).

**Pima Indians Diabetes:** contém 768 registros, 8 atributos preditivos e duas classes relacionadas à presença ou ausência de diabetes (500 negativos e 268 positivos).

## Metodologia

O projeto utiliza uma rede neural MLP cujos pesos são otimizados por meio do algoritmo BBO.

Foram avaliadas as seguintes configurações:

- BBO padrão, sem SMOTE.
- BBO padrão, com SMOTE.
- BBO exponencial, sem SMOTE.
- BBO exponencial, com SMOTE.

Os experimentos foram realizados nas duas bases de dados, utilizando validação cruzada estratificada com cinco divisões (5-fold).

No total, foram realizadas **40 execuções de treinamento e avaliação**, além de 8 execuções complementares para análise de convergência.

O pré-processamento inclui tratamento de dados ausentes e padronização dos atributos. Quando utilizado, o SMOTE é aplicado somente aos dados de treinamento, evitando vazamento de informações para os dados de avaliação.

### Configuração dos algoritmos

- População do BBO: 15 indivíduos.
- Número de gerações: 30.
- Taxas de migração: modelo linear de Simon (2008), em que habitats com melhor fitness têm maior emigração e menor imigração.
- Probabilidade máxima de mutação: 0,005.
- Arquitetura da MLP: uma camada oculta com `2N + 1` neurônios, sendo `N` o número de atributos de entrada.
- Função de avaliação do treinamento: erro quadrático médio (MSE).
- Variante exponencial: parâmetro λ = 0,2.

## Métricas de avaliação

Foram utilizadas as seguintes métricas:

- Acurácia
- Precisão
- Sensibilidade
- Especificidade
- F1-score
- G-mean
- ROC-AUC
- Erro quadrático médio (MSE)
- Tempo de execução

Também foram geradas matrizes de confusão e curvas de convergência.

## Resultados

Médias dos cinco folds (`results/metrics/summary_results.csv`). Em negrito, o melhor valor de cada métrica em cada base.

| Base de dados | Algoritmo | SMOTE | Acurácia | Sensibilidade | Especificidade | F1-score | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | BBO padrão | Não | 97,19% | 94,82% | **98,61%** | 96,15% | 99,39% |
| Breast Cancer | BBO padrão | Sim | 95,61% | 93,42% | 96,92% | 94,03% | 99,44% |
| Breast Cancer | BBO exponencial | Não | **97,37%** | **96,22%** | 98,04% | **96,45%** | 99,42% |
| Breast Cancer | BBO exponencial | Sim | 97,19% | 95,78% | 98,05% | 96,22% | **99,54%** |
| Diabetes | BBO padrão | Não | 75,78% | 58,97% | 84,80% | 62,96% | 83,05% |
| Diabetes | BBO padrão | Sim | 73,17% | **76,48%** | 71,40% | **66,81%** | 81,94% |
| Diabetes | BBO exponencial | Não | **77,08%** | 57,08% | **87,80%** | 63,47% | **83,19%** |
| Diabetes | BBO exponencial | Sim | 73,43% | 74,97% | 72,60% | 66,43% | 81,63% |

Principais observações:

- O BBO exponencial sem SMOTE obteve a maior acurácia média nas duas bases: 97,37% no câncer de mama e 77,08% em diabetes.
- Na base de diabetes, o SMOTE aumentou a sensibilidade (de 57,08% para 74,97% no BBO exponencial e de 58,97% para 76,48% no BBO padrão) e o F1-score, ao custo de menor especificidade e acurácia.
- Na base de câncer de mama, cujas classes já são bem separadas, o SMOTE não trouxe ganho relevante.
- As curvas de convergência mostram redução contínua do MSE de treinamento ao longo das 30 gerações, ainda em queda ao final, o que sugere que um número maior de gerações pode melhorar a otimização.

Os resultados são descritivos: não foi aplicado teste de significância estatística, e cinco folds não equivalem às 30 execuções independentes do artigo. Também não representam uma reprodução exata do artigo original, mas uma investigação experimental baseada em parte de sua metodologia.

## Estrutura do projeto

```text
mini-projeto-ia/
├── data/
│   └── raw/
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── mlp_model.py
│   ├── bbo_optimizer.py
│   ├── evaluation.py
│   ├── experiments.py
│   ├── visualization.py
│   └── convergence.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Como executar

É necessário possuir o Python instalado.

Clone o repositório:

```bash
git clone https://github.com/GustavoSilva19/mini-projeto-ia.git
cd mini-projeto-ia
```

Crie e ative um ambiente virtual no Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Execute os experimentos:

```powershell
python main.py
```

Os resultados são armazenados em `results/metrics/`, enquanto os gráficos ficam em `results/figures/`.

## Conclusão

O mini-projeto permitiu analisar a aplicação da Otimização Baseada em Biogeografia no treinamento de redes neurais para classificação de dados médicos.

A MLP treinada pelo BBO alcançou 97,37% de acurácia média na base de câncer de mama e 77,08% na base de diabetes, ambos com a variante exponencial sem SMOTE. As diferenças entre o BBO padrão e o exponencial foram pequenas, o que é coerente com o efeito sutil da distribuição exponencial com λ = 0,2 sobre a probabilidade de migração.

O SMOTE mostrou-se útil quando o objetivo é identificar mais casos positivos na base de diabetes, mas reduz a especificidade, evidenciando a importância de avaliar métricas além da acurácia em problemas médicos desbalanceados.

O estudo também possibilitou aplicar conceitos de otimização, aprendizado de máquina, validação cruzada e avaliação de classificadores.

## Referências

- LI, X.-D. et al. Multi-layer perceptron classification method of medical data based on biogeography-based optimization algorithm with probability distributions. *Applied Soft Computing*, v. 121, 108766, 2022.
- SIMON, D. Biogeography-based optimization. *IEEE Transactions on Evolutionary Computation*, v. 12, n. 6, p. 702–713, 2008.
- MIRJALILI, S.; MIRJALILI, S. M.; LEWIS, A. Let a biogeography-based optimizer train your multi-layer perceptron. *Information Sciences*, v. 269, p. 188–209, 2014.
- CHAWLA, N. V. et al. SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, v. 16, p. 321–357, 2002.