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

**Autores:** Xu-Dong Li e colaboradores.

**Periódico:** Applied Soft Computing, 2022.

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

**Breast Cancer Wisconsin Diagnostic (WDBC):** contém 569 registros, 30 atributos preditivos e duas classes relacionadas ao diagnóstico de tumores.

**Pima Indians Diabetes:** contém 768 registros, 8 atributos preditivos e duas classes relacionadas à presença ou ausência de diabetes.

## Metodologia

O projeto utiliza uma rede neural MLP cujos pesos são otimizados por meio do algoritmo BBO.

Foram avaliadas as seguintes configurações:

- BBO padrão, sem SMOTE.
- BBO padrão, com SMOTE.
- BBO exponencial, sem SMOTE.
- BBO exponencial, com SMOTE.

Os experimentos foram realizados nas duas bases de dados, utilizando validação cruzada estratificada com cinco divisões (5-fold).

No total, foram realizadas **40 execuções de treinamento e avaliação**, além de experimentos complementares para análise de convergência.

O pré-processamento inclui tratamento de dados ausentes e padronização dos atributos. Quando utilizado, o SMOTE é aplicado somente aos dados de treinamento, evitando vazamento de informações para os dados de avaliação.

### Configuração dos algoritmos

- População do BBO: 15 indivíduos.
- Número de gerações: 30.
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

Entre as configurações avaliadas, o BBO exponencial com SMOTE apresentou os melhores valores médios de acurácia e F1-score nas duas bases.

| Base de dados | Acurácia | F1-score | ROC-AUC |
|---|---:|---:|---:|
| Breast Cancer | 85,95% | 83,27% | 95,57% |
| Diabetes | 69,80% | 54,72% | 72,74% |

A utilização do SMOTE melhorou o F1-score da configuração exponencial em ambas as bases, embora seus efeitos variem conforme a métrica e o algoritmo utilizado.

Os resultados não representam uma reprodução exata do artigo original, mas uma investigação experimental baseada em parte de sua metodologia.

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

Os experimentos demonstraram diferenças de desempenho entre as configurações padrão e exponencial do BBO, além de evidenciar a importância de avaliar técnicas de balanceamento, como o SMOTE, especialmente em métricas relacionadas à identificação das classes.

O estudo também possibilitou aplicar conceitos de otimização, aprendizado de máquina, validação cruzada e avaliação de classificadores.
