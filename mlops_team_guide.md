# Guia de Equipa — Projeto MLOps
### Mortgage Default Prediction | Nova IMS 2026

---

## 1. Objetivo do Projeto

Construir um pipeline MLOps completo para prever incumprimento de crédito hipotecário, usando o dataset Freddie Mac (2005).

A narrativa central é o **"Big Short"** — demonstrar que um sistema MLOps com deteção de drift teria identificado os sinais da crise de 2008 com antecedência.

**Métricas de sucesso:**
- Métrica primária: AUC-ROC
- Métricas secundárias: F1, Recall
- O dataset tem ~18:1 de desequilíbrio de classes (~5.4% de defaults)

---

## 2. Os Dados

O dataset tem dois ficheiros que se relacionam pelo `loan_sequence_number`:

| Ficheiro | Linhas | O que contém |
|---|---|---|
| `sample_orig_2005.txt` | ~50k | 1 linha por empréstimo — características no momento da criação (FICO, LTV, taxa juro…) |
| `sample_svcg_2005.txt` | ~3.87M | N linhas por empréstimo — histórico mensal (pagamentos, incumprimento…) |

**Variável target:** vem do `zero_balance_code` no ficheiro de performance.
- Códigos 2, 3, 9 → `default = 1`
- Qualquer outro → `default = 0`

O modelo aprende com as características da **originação** a prever se o empréstimo vai entrar em default — por isso fazemos um join entre os dois ficheiros.

---

## 3. Pipeline Completo

```
data_ingestion → data_quality → data_cleaning → feat_engineering → model_train → model_selection → model_predict → data_drift
```

| Pipeline | Ferramenta principal | Estado |
|---|---|---|
| `data_ingestion` | Pandas | 🟡 Incompleto — falta performance + join |
| `data_quality` | Great Expectations | ⬜ A fazer |
| `data_cleaning` | Pandas | ⬜ A fazer |
| `data_feat_engineering` | Pandas + Scikit-learn | ⬜ A fazer |
| `model_train` | MLflow | ⬜ A fazer |
| `model_selection` | MLflow + SHAP | ⬜ A fazer |
| `model_predict` | FastAPI + Docker | ⬜ A fazer |
| `data_drift` | Evidently | ⬜ A fazer |

---

## 4. Divisão de Tarefas

| Membro | Pipelines | Dificuldade |
|---|---|---|
| Membro 1 | `data_quality`, `data_cleaning`, EDA, testes de dados | Média |
| Membro 2 | `feat_engineering`, `data_split`, Feature Store (Hopsworks), testes de features | Média/Difícil |
| Membro 3 | `model_train` + MLflow, `model_selection`, SHAP, testes de pipeline | Difícil |
| Membro 4 | `model_predict` + FastAPI + Docker, `data_drift` + Evidently, relatório | Difícil |

---

## 5. Como o Kedro Funciona

O Kedro organiza o código em camadas. Para cada pipeline que criares, vais sempre tocar nos mesmos 3 ficheiros:

### 5.1 Os ficheiros que **tu** escreves

**`nodes.py`** — a lógica pura
```
src/mortgage_default/pipelines/<pipeline>/nodes.py
```
Funções Python normais. Recebem DataFrames, devolvem DataFrames. Não sabem nada do Kedro — são testáveis de forma isolada.

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["credit_score"])
    return df
```

**`pipeline.py`** — a sequência
```
src/mortgage_default/pipelines/<pipeline>/pipeline.py
```
Liga as funções do `nodes.py` em sequência, definindo o que entra e o que sai de cada uma.

```python
Node(
    func=clean_data,
    inputs="origination_data_named",   # nome no catalog.yml
    outputs="cleaned_data",            # nome no catalog.yml
    name="clean_data_node"
)
```

**`catalog.yml`** — os dados
```
conf/base/catalog.yml
```
Liga os nomes usados no `pipeline.py` a ficheiros reais no disco.

```yaml
cleaned_data:
  type: pandas.ParquetDataset
  filepath: data/02_intermediate/cleaned_data.parquet
```

### 5.2 O ficheiro que o Kedro gere sozinho

**`pipeline_registry.py`** — não precisas de tocar aqui.
O `find_pipelines()` descobre automaticamente todos os pipelines que criares, desde que sigam a estrutura de pastas correta.

### 5.3 Regra de ouro

> **O nome no `inputs`/`outputs` do `pipeline.py` tem de existir no `catalog.yml`.**
> É esta correspondência que liga tudo.

---

## 6. Comandos Essenciais

```bash
# Ativar o ambiente (dentro da pasta mortgage-default)
source .venv/Scripts/activate

# Ver os pipelines registados
kedro registry list

# Correr um pipeline específico
kedro run --pipeline data_ingestion

# Correr o pipeline completo
kedro run

# Correr os testes
pytest tests/

# Adicionar um pacote novo
uv add <nome_do_pacote>

# Sincronizar o ambiente após git pull
uv sync
```

---

## 7. Boas Práticas e Reprodutibilidade

### Git
- **Nunca** trabalhar diretamente em `main`
- Cada membro cria uma branch por feature: `git checkout -b feat/data-quality`
- Fazer Pull Request para `develop` quando a feature estiver completa
- Só o Miguel (ou quem for designado) faz merge de `develop` para `main`

### Ambiente
- Quando adicionares um pacote novo: `uv add <pacote>` — isto atualiza o `pyproject.toml` e o `uv.lock`
- Fazer sempre commit do `uv.lock` para o repo — é o que garante que todos têm as mesmas versões
- Quando fizeres `git pull` e o `uv.lock` tiver mudado: correr `uv sync` antes de qualquer outra coisa

### Dados
- Os ficheiros raw (`data/01_raw/`) **nunca** vão para o Git — estão no `.gitignore`
- Os ficheiros intermédios (`data/02_intermediate/`, etc.) também não
- O que vai para o Git é o **código** que gera esses ficheiros, não os ficheiros em si
- Cada membro partilha os dados via outro canal (pasta partilhada, link Freddie Mac)

### Código
- A lógica vive nos `nodes.py` — funções puras, sem side effects
- Nunca guardar ficheiros dentro das funções — o Kedro trata disso via `catalog.yml`
- Cada função deve ter docstring com Args e Returns (já está no padrão do `data_ingestion`)
- Escrever testes para todas as funções do `nodes.py`

### Estrutura de pastas dos dados
```
data/
├── 01_raw/          ← ficheiros originais, nunca modificar
├── 02_intermediate/ ← outputs de data_ingestion, data_quality, data_cleaning
├── 03_primary/      ← outputs de feat_engineering
├── 04_feature/      ← features prontas para o modelo
├── 05_model_input/  ← train/test splits
├── 06_models/       ← modelos treinados
├── 07_model_output/ ← predições
└── 08_reporting/    ← relatórios, gráficos SHAP
```

---

## 8. Como Criar um Pipeline Novo

```bash
# 1. Criar a estrutura de pastas automaticamente
kedro pipeline create <nome_do_pipeline>

# 2. Escrever as funções em nodes.py

# 3. Ligar as funções em pipeline.py

# 4. Registar os inputs/outputs em catalog.yml

# 5. Verificar que o Kedro descobriu o pipeline
kedro registry list

# 6. Testar
kedro run --pipeline <nome_do_pipeline>
```

---

*Documento gerado em junho 2026 | Projeto MLOps — Nova IMS*
