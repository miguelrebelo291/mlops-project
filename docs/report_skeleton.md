# MLOps Project — Report (skeleton)

> **Constraints from the handout:** max **6 pages**. Graded on report quality, code, and
> creativity in using the technologies. Fill the `TODO` / _[in brackets]_ placeholders.
> Suggested page budget is in each section header — adjust as needed.
>
> **Delete this quote block and all `> notes` before submitting.**

---

## 0. Header _(not counted)_
- **Title:** Mortgage Default Prediction — an MLOps proof of concept
- **Course / Year:** Nova IMS — MLOps 2026
- **Team:** _[names + student numbers]_
- **Repository:** _[Git link]_  ·  **Data sample:** _[link or “in zip”]_

---

## 1. Introduction — Data, Goal & Success Metrics _(~0.75 page)_
> Maps to handout bullet 1: *“why you chose that data and what you try to achieve. Define your success metrics.”*

- **Problem & motivation.** Predict mortgage default at loan origination. Narrative angle:
  the *“Big Short”* — an MLOps system with drift detection would have surfaced the 2008 signals early.
- **Data.** Freddie Mac Single-Family Loan-Level dataset (2005 sample):
  - `sample_orig_2005.txt` (~50k loans, origination features) + `sample_svcg_2005.txt` (~3.87M monthly performance rows), joined on `loan_sequence_number`.
  - **Target:** `default = 1` when `zero_balance_code ∈ {2,3,9}`, else `0`.
- **Why this data.** _[real-world, well-documented, class imbalance, temporal angle for drift]_
- **Success metrics.** Primary: **AUC-ROC**. Secondary: **F1, Recall**. Note **~18:1 class imbalance (~5.4% defaults)** and why accuracy is the wrong metric here.

---

## 2. Project Planning _(~0.5 page)_
> Maps to handout bullet 2: *“how you organized and scheduled the different steps (agile/sprints).”*

- **Team roles** _(from team guide)_: Member 1 — data_quality/cleaning/EDA; Member 2 — feature engineering, data_split, feature store; Member 3 — model train/selection + SHAP; Member 4 — serving + drift + report.
- **Sprints / schedule.** _[table: sprint → goals → deliverables. Mention Git workflow: feature branches → develop → main]_
- _Optional:_ small Gantt or sprint table screenshot.

---

## 3. Pipeline Architecture _(~1 page)_
> Shows the modular Kedro design and the components the handout asks for. A diagram earns “creativity” points.

- **Diagram.** `data_ingestion → data_quality → data_cleaning → feature_engineering → model_train → model_predict → data_drift` _[insert kedro-viz screenshot]_
- **Training vs Inference pipelines** (leakage-safe): shared stages, but transformers are
  *fit* only on train and *reused* at inference. _[short paragraph]_
- **Component checklist** (handout “components” list) — keep a small status table:

  | Component | Tool | Status |
  |---|---|---|
  | Data unit tests / quality | Great Expectations | _[done?]_ |
  | Feature store | Hopsworks | ✅ done |
  | Experiment & model versioning | MLflow | _[done?]_ |
  | Explainability (SHAP) | SHAP | TODO |
  | Model serving + containers | FastAPI + Docker | TODO |
  | Data drift | Evidently | TODO |
  | Function/pipeline tests | pytest | partial _[count]_ |

- **Orchestration.** _(handout p.2 stresses this — give it its own short paragraph.)_
  The project is organized as **independent, composable Kedro pipelines**:
  - **Full sequential run** — the whole flow end to end:
    `data_quality → data_cleaning → feature_engineering → model_train → data_drift`
    (`kedro run`, or chained via the pipeline registry).
  - **Run a single pipeline in isolation** — e.g. once the model is in production you may
    run only `data_quality` on incoming data, or only `data_drift` on a sample
    (`kedro run --pipeline data_drift`). _[name the commands you actually use]_
  - This mirrors a real MLOps setup: training and monitoring are decoupled, so each
    stage can run on its own schedule/trigger without re-running the others.

---

## 4. Results & Conclusions _(~1.75 pages — the core)_
> Maps to handout bullet 3: *“results and conclusions from data exploration and data modelling (plots, feature importance, explainability).”*

- **4.1 Data exploration (EDA).** _[2–3 key plots: target imbalance, FICO/LTV/DTI distributions, default rate by feature]_
- **4.2 Feature engineering.** Drop low-value cols (`seller_name`, `servicer_name`, `msa`),
  dates→numeric, one-hot, leakage-safe imputation. _[note nº of final features]_
- **4.3 Modelling.** _[models tried, train/test split, hyperparams]_
- **4.4 Performance.** _[table: AUC-ROC / F1 / Recall per model; ROC curve; confusion matrix]_
- **4.5 Explainability (SHAP).** _[SHAP summary plot + 2–3 sentences on top drivers of default]_
- **4.6 Drift experiment.** _[Evidently report on a drifted sample; what metrics moved]_
- **Conclusions.** _[what worked, what the model says about default risk, the “Big Short” payoff]_

---

## 5. From Proof of Concept to Production _(~1 page)_
> Maps to handout bullet 4: *advantages of the tech, risks and mitigations.* Use the handout’s own example style.

- **Architecture in production.** _[batch vs online, where the feature store/MLflow registry/serving sit]_
- **Advantages of the stack.** Kedro (modularity/repro), MLflow (versioning), Hopsworks (feature reuse train↔serve), Evidently (drift), Docker (portability).
- **Risks & mitigations** _(table)_:
  - *“We use only Pandas — at scale the pipeline won’t be efficient. Mitigation: +X weeks to port to Spark.”*
  - Data drift over time → automated drift alerts + retraining trigger.
  - Class imbalance / threshold choice → calibrate, monitor recall in production.
  - _[add 1–2 more]_

---

## 6. Packages & Versions _(~0.25 page)_
> Maps to handout bullet 5. Generate the real list from your env (don’t handwrite):
> `uv pip list` or `uv export --format requirements-txt`. Keep only the key ones.

- kedro, kedro-datasets, pandas, scikit-learn, great-expectations, mlflow, hopsworks,
  hops-deltalake, evidently, fastapi, shap … _[+ versions]_

---

## Appendix _(optional, may not count toward 6 pages — check with prof)_
- Repo structure, how to run (`kedro run --pipeline <name>`), reproducibility notes.

---

### Page-budget summary (target ≈ 6 pages)
| Section | Pages |
|---|---|
| 1 Introduction | 0.75 |
| 2 Planning | 0.5 |
| 3 Architecture | 1.0 |
| 4 Results | 1.75 |
| 5 Production | 1.0 |
| 6 Packages | 0.25 |
| _buffer/figures_ | 0.75 |
