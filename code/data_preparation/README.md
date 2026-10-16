# R2R thesis – data preparation pipeline (Section 3.1)

This folder turns the output of phase 1, the final task register, into the inputs of the analysis pipeline in `../final_pipeline/`.

```
task categorization/  (phase 1)  ->  final_task_register.csv
code/data_preparation/ (this)    ->  task_master.csv, final_aei_panel.csv, elicited_time_fractions.csv
code/final_pipeline/  (analysis) ->  every figure and table in the thesis
```

## Setup

1. Requirements: `pandas`, `numpy`, `scipy`, `scikit-learn`, `openpyxl`, `krippendorff`, `sentence-transformers` (needs `torch`).
   - Only notebook 02 needs `sentence-transformers`, and only if its embedding cache is missing. The first encode downloads `all-mpnet-base-v2` (about 440 MB).
   - Notebook 01 needs `anthropic`, `openai` and `google-genai` only if a model reply is missing from the cache.
2. Open Jupyter **in this folder**, so the notebooks can `import prep_common`.
3. Run `python download_aei.py` once to fetch the raw AEI files (Releases 1–5, about 310 MB, checksum-verified) into `inputs/aei/`. All other raw inputs are already in `inputs/`. Nothing in `inputs/` is ever written to.

## Run order

| # | Notebook | Thesis | Reads | Writes (`outputs/…`) | Hands off to `final_pipeline/data/` |
|---|---|---|---|---|---|
| 01 | `01_time_fraction_elicitation.ipynb` | 3.1.1, App. C.1, Table D6 | register, O\*NET tasks, reply cache | `01_time_fraction/` | `elicited_time_fractions.csv` |
| 02 | `02_ridge_task_expertise.ipynb` | 3.1.2, Table D5 | register, O\*NET tasks, BLS education | `02_ridge_expertise/` | – |
| 03 | `03_task_master.ipynb` | 3.1.3, Table B1 | register + 01 + 02 + OEWS, O\*NET ratings, Eloundou β | `03_task_master/` | `task_master.csv`, `final_task_register.csv` |
| 04 | `04_aei_panel.ipynb` | 3.1.4, App. B | 03 + AEI releases | `04_aei_panel/` | `final_aei_panel.csv` |

01 and 02 are independent of each other. 03 needs both, and 04 needs 03. After 04, run `final_pipeline` 01–05 in that folder's order.

**Runtime.** 01, 03 and 04 take under a minute each. On first run, 02 spends 10–20 minutes encoding 18,005 task statements on a CPU. The embeddings are then cached in `outputs/02_ridge_expertise/embeddings/` and reused, so later runs take a few minutes (mostly the 200-sample bootstrap).

**No API keys needed.** Every model reply from the original elicitation is in `inputs/time_fraction/elicitation_cache.json`, so 01 rebuilds the time fractions exactly. API keys are read only from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) and only when a reply is missing. They are never stored in a notebook.

`prep_common.py` holds everything shared:

- paths and the output layout
- the task-id and task-text normalisation rules
- the register and O\*NET loaders
- `handoff()`, which copies finished files into `../final_pipeline/data/`

Environment overrides: `R2R_PREP_INPUT_DIR`, `R2R_PREP_OUT_DIR`, and `R2R_DATA_DIR` (the hand-off target, shared with `final_pipeline`).

## Inputs (`inputs/`)

| File | Source | Used by |
|---|---|---|
| `final_task_register.csv` | phase 1 output (`task categorization/`; Appendix A, Table A2) | 01, 02, 03 |
| `onet/task_statements.xlsx` | O\*NET Task Statements | 01, 02, 03 |
| `onet/task_ratings.xlsx`, `onet/task_categories.xlsx` | O\*NET Task Ratings, FT (frequency) scale and its category labels | 03 |
| `bls/BLS_education.xlsx` | BLS Employment Projections Table 5.4 (typical education for entry) | 02 |
| `bls/OEWS_Wage_May2025.xlsx` | BLS OEWS national estimates, May 2025 | 03 |
| `eloundou/eloundou_beta.tsv` | Eloundou et al. (2024) task-level exposure | 03 |
| `time_fraction/elicitation_cache.json` | raw model replies, keyed `model‖SOC‖title` (30 = 10 occupations × 3 models) | 01 |
| `aei/release_1` … `release_5` | AEI raw files as published | 04 |
| `aei/release_6/` | empty; drop the R6 `aei_claude_ai_*.csv` / `aei_1p_api_*.csv` files here to add Release 6 | 04 |

## Output layout

Each notebook writes into its own folder. Every table is a CSV, ready for charts or slides.

| Folder | Contents |
|---|---|
| `outputs/01_time_fraction/` | `elicited_time_fractions.csv` (212 tasks: per-model hours, 3-model mean, SD, range, CV); `model_hours_long.csv` (every reply, one row per model × task); `agreement_summary.csv` (Krippendorff's α, ordinal, for all tasks and for in-scope tasks); `agreement_report.txt` |
| `outputs/02_ridge_expertise/` | `ridge_task_complexity.csv` (212 tasks: predicted years, BLS years, bootstrap SD and 95% CI); `tableD5_ridge_diagnostics.csv`; `validation_summary.json`; `alpha_sensitivity.csv`; `cv_fold_metrics.csv`; `training_pool.csv`; `r2r_holdout_tasks.csv`; `embeddings/` (cache) |
| `outputs/03_task_master/` | `task_master.csv` (Table B1 schema); `task_master_checks.csv` (missing values and ranges per column); `subprocess_summary.csv` (in-scope tasks per subprocess, mean wage, ridge years and β, total weekly hours) |
| `outputs/04_aei_panel/` | `final_aei_panel.csv` (Table B2 schema); `anthropic_panel_onet_global.csv` (pre-refinement canonical panel); `build_log.csv` (rows per release: loaded, then task-matched, then in scope); `pct_rescaled_groups.csv`; `panel_coverage.csv` (Table B3) |

## Method notes

| Notebook | Note |
|---|---|
| 01 | API clients are created only when a reply is missing from the cache, and keys are read from the environment. `cross_model_mean_hours` is not renormalised here; the analysis renormalises it within each scope at the point of use. |
| 02 | Ridge is fit on all O\*NET tasks outside the 10 study occupations and predicts all 212 register tasks. The random KFold and the occupation bootstrap use seed 42, the random held-out-domain draw seed 43. |
| 03 | Task frequency converts the O\*NET FT categories to occurrences per year (1, 3.5, 25, 114, 250, 750, 2000 for categories 1–7) and takes the share-weighted mean; 9 tasks without FT ratings are left blank. Frequency is not used in the final constructs. |
| 04 | Build, refine and pct normalisation run as three steps in one notebook. Release 1 and 2 files carry no collection window, so their dates are set in `RELEASE_DATES`. Dates are dropped from the final panel. |

## Status

Run end to end on 16 October 2026 (Python 3.12, Windows, scikit-learn 1.7.2), from the raw inputs through `final_pipeline` 01–05, with no errors. The values reported in the thesis (Section 3.1.2, Tables 7, D5 and D8) match this run.

**Reproducibility.** Use scikit-learn 1.7.2 (`pip install scikit-learn==1.7.2`). `GroupKFold` breaks ties between equally sized occupations differently across scikit-learn and numpy versions and CPUs, so the grouped-CV diagnostics (R², RMSE, Spearman ρ, leakage gap) can shift in the second or third decimal on another setup. The ridge predictions themselves don't depend on the folds, because the final model is fit on the whole training pool; they can differ only in the fifth decimal (float32 arithmetic).

## Sources and references

Full references, licences and the O\*NET attribution statement are in the [root README](../../README.md#references).

| Notebook | Method / data it follows |
|---|---|
| 01 time fraction | prompt and design adapted from Tamkin & McCrory (2025); O\*NET 30.2 task lists (National Center for O\*NET Development, 2026); agreement as Krippendorff's α (Hayes & Krippendorff, 2007) |
| 02 ridge expertise | BLS typical education for entry (U.S. Bureau of Labor Statistics, 2026a); `all-mpnet-base-v2` embeddings (Reimers & Gurevych, 2019; Song et al., 2020); Ridge, GroupKFold (Pedregosa et al., 2011); expertise framing from Autor & Thompson (2025) |
| 03 task master | OEWS May 2025 wages (U.S. Bureau of Labor Statistics, 2026b); O\*NET task ratings (FT scale); β from Eloundou et al. (2024), `data/full_labelset.tsv` of [openai/GPTs-are-GPTs](https://github.com/openai/GPTs-are-GPTs) |
| 04 AEI panel | Anthropic Economic Index Releases 1–5 (Anthropic, 2026a; Handa et al., 2025; Appel et al., 2025, 2026; Massenkoff et al., 2026), [huggingface.co/datasets/Anthropic/EconomicIndex](https://huggingface.co/datasets/Anthropic/EconomicIndex), built on Clio (Tamkin et al., 2024) |
