
# ⛽ Benchmark de Previsão do Preço da Gasolina no Brasil

## Comparação entre Modelos Estatísticos, Machine Learning e Deep Learning para Previsão Recursiva (Brasil × Minas Gerais)

---

# 📄 Resumo

Este trabalho apresenta um benchmark comparativo entre sete modelos preditivos aplicados à previsão recursiva do preço da gasolina no Brasil entre 2013 e 2025.

O estudo compara o desempenho dos modelos em duas escalas geográficas:

- Brasil (agregado nacional)
- Minas Gerais (nível estadual)

---

# 🎯 Modelos Avaliados

## Estatísticos Clássicos

- Naive
- SARIMAX
- Exponential Smoothing
- THETA
- ETS_no_trend

## Machine Learning

- XGBoost

## Deep Learning

- LSTM (janela temporal de 24 observações)

---

# 🏆 Principais Resultados

## Brasil

- Melhor modelo: SARIMAX
- MAPE: 0,72%
- MAE: R$ 0,0450 por litro

## Minas Gerais

- Melhor modelo: ETS_no_trend
- MAPE: 0,68%
- MAE: R$ 0,0420 por litro

---

# 🔍 Principais Descobertas

- Confirmação prática do Teorema No-Free-Lunch.
- O melhor modelo depende da escala geográfica analisada.
- A variável ICMS reduziu o erro em até 0,4 pontos percentuais.
- A LSTM corrigida alcançou o segundo melhor desempenho no cenário nacional.
- O petróleo Brent foi a variável mais importante no modelo nacional.
- Em Minas Gerais, o preço passado foi a variável dominante.

---

# 📥 Dados

Fonte oficial da Agência Nacional do Petróleo (ANP):

https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis

## Instruções

1. Baixe os arquivos CSV semestrais de 2013 a 2025.
2. Crie a pasta:

data/raw/

3. Copie todos os arquivos CSV para essa pasta.

---

# ⚙️ Requisitos

- Python 3.10 ou superior

Verificar instalação:

```bash
python --version
```

---

# 🚀 Instalação

## Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# ▶️ Execução do Benchmark

```bash
python bench17_final_completo.py
```

O script executa:

- Treinamento dos sete modelos
- Forecast recursivo para 2025
- Cálculo das métricas
- Geração de gráficos
- Exportação de CSVs
- Geração de relatórios

---

# 📈 Gráficos Adicionais

```bash
python graficos_previsoes.py
```

Serão produzidos:

- Top 3 modelos nacionais
- Top 3 modelos de Minas Gerais
- Comparação dos vencedores
- Análise dos resíduos

---

# 📁 Estrutura do Projeto

```text
gasolina-benchmark/
│
├── bench17_final_completo.py
├── graficos_previsoes.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│
├── results/
│   ├── CSVs/
│   ├── Graficos/
│   └── Relatorios/
│
└── my_results/
    ├── CSVs/
    ├── Graficos/
    └── Relatorios/
```

---

# 📊 Arquivos Gerados

## Métricas

- metrics_Nacional.csv
- metrics_MG.csv

## Forecasts

- forecasts_Nacional.csv
- forecasts_MG.csv

## Resíduos

- residuos_Nacional.csv
- residuos_MG.csv

## Importância das Variáveis

- feature_importance_Nacional.csv
- feature_importance_MG.csv

## Sensibilidade

- sensibilidade_resultados.csv

---

# 🖼️ Gráficos Produzidos

- 01_comparativo_nacional_mg.png
- 02_ranking_modelos.png
- 03_heatmap_metricas.png
- 04_analise_residuos_Nacional.png
- 04_analise_residuos_MG.png
- 05_feature_importance_Nacional.png
- 05_feature_importance_MG.png
- 06_sensibilidade_impacto.png
- nacional_top3_previsoes.png
- mg_top3_previsoes.png
- comparativo_melhores_modelos.png
- erros_melhores_modelos.png

---

# 🔬 Tecnologias Utilizadas

- Python 3.10+
- pandas 2.0.3
- numpy 1.25.2
- statsmodels 0.14.0
- xgboost 3.2.0
- tensorflow-cpu 2.10.0
- yfinance 1.4.1
- scikit-learn 1.3.2
- matplotlib 3.8.2
- seaborn 0.13.0
- joblib 1.3.2

---

# 📝 Metodologia

## Forecast Recursivo

1. Treinamento com dados de 2013–2024.
2. Previsão do primeiro período de 2025.
3. Uso da previsão anterior como entrada para a próxima previsão.
4. Repetição até completar os 24 períodos quinzenais de 2025.

## Divisão Cronológica

Treino:
- 2013–2022
- 240 observações

Validação:
- 2023–2024
- 48 observações

Teste:
- 2025
- 24 observações

Essa estratégia elimina vazamento de informação futura.

---

# 📏 Métricas de Avaliação

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)

---

# 👨‍💻 Autor

Cristiano Márcio de Araújo Almeida

Programa de Pós-Graduação em Modelagem Computacional

Universidade Federal de Juiz de Fora (UFJF)

E-mail: cristiano.almeida@estudante.ufjf.br

---

# 📚 Citação

```bibtex
@article{almeida2026benchmark,
  title={Benchmark de Previsão do Preço da Gasolina no Brasil},
  author={Almeida, Cristiano Márcio de Araújo},
  journal={Programa de Pós-Graduação em Modelagem Computacional},
  institution={Universidade Federal de Juiz de Fora},
  year={2026}
}
```

---

# 📜 Licença

Este projeto está licenciado sob a Licença MIT.

---

# 🙏 Agradecimentos

À Agência Nacional do Petróleo (ANP) pela disponibilização pública dos dados utilizados neste estudo.

À Universidade Federal de Juiz de Fora (UFJF) pelo suporte acadêmico e institucional.
