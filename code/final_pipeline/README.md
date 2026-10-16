# R2R thesis – analysis pipeline (Sections 3.2–4.2, Appendix D)

This folder builds every construct, figure and table of the thesis from two inputs: the task master and the long-format AEI panel. Both are produced by `../data_preparation/`.

## Setup

1. The inputs in `data/` are written by the data-preparation pipeline (`../data_preparation/`, run its notebooks 01–04 first):
   - `data/task_master.csv`
   - `data/final_aei_panel.csv`
   - `data/elicited_time_fractions.csv` and `data/final_task_register.csv` (notebook 05 only)

   Alternatively, keep them elsewhere and set the environment variable `R2R_DATA_DIR` before starting Jupyter.
2. Optional: put the Inter `.ttf` files in `fonts/` to draw the charts in the thesis typeface. Otherwise they fall back to Helvetica, Arial or DejaVu.
3. Requirements: see `../../requirements.txt`. This folder needs `pandas`, `numpy`, `matplotlib`, `scipy`, plus `krippendorff` for notebook 05.
4. Open Jupyter **in this folder** so the notebooks can `import r2r_common`.

## Run order

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 01 | `01_constructs_and_checks.ipynb` | `data/*.csv` | `outputs/0_constructs/*.csv` (all intermediate constructs) and the data checks |
| 02 | `02_descriptive_statistics.ipynb` | data + constructs | Tables 2, 3, D3, D4, D9 and Figure D3 |
| 03 | `03_RQ1_location_of_exposure.ipynb` | data + constructs | Figures 1–10 and D1 |
| 04 | `04_RQ2_expertise_and_effectiveness.ipynb` | data + constructs | Figures 11–26, D2, D4, D5 and Tables 4–8, D1, D2, D7, D8, D10 |
| 05 | `05_appendix_time_fraction_agreement.ipynb` | `data/elicited_time_fractions.csv`, `data/final_task_register.csv` | Table D6 (independent of 01–04) |

Run 01 first. 02–04 can then run in any order, because they only read `outputs/0_constructs/`.

`r2r_common.py` holds everything shared, so each construct is defined once:

- paths
- the data loader and panel accessors
- the construct builders: observed exposure, concentration, task success / effective exposure, the coverage-gap decomposition, the net skill-coverage gap, and the R2R-relative indices
- the regression helpers: conventional and task-clustered standard errors, F-tests, and thesis-style tables
- the chart house style

## Output layout

Outputs are grouped by **where they sit in the thesis**:

| Folder | Contents |
|---|---|
| `outputs/0_constructs/` | intermediate CSVs that later notebooks read back in (`task_exposure`, `subprocess_exposure`, `process_exposure`, `task_dispersion`, `subprocess_concentration`, `process_concentration`, `task_success`, `effective_exposure`, `r2r_indices`, plus the panel coverage checks) |
| `outputs/1_ch3_descriptives/` | Tables 2 and 3, and `section3_3_facts.txt` |
| `outputs/2_ch4_1_rq1_location/` | Figures 1–10, their data CSVs, and `fit_statistics.csv` |
| `outputs/3_ch4_2_rq2_expertise/` | Figures 11–26, Tables 4–8, `text_values_4_2_2.csv`, and `fit_statistics.csv` |
| `outputs/4_appendix_D/` | every Appendix D figure and table |

File names start with the figure or table number. For two-panel figures, `a` is the **left** panel and `b` the **right** panel as placed in the thesis (e.g. `fig19a_..._R4.png` left, `fig19b_..._R5.png` right). `fit_statistics.csv` lists every slope, intercept, R² and n drawn on a chart, so the numbers quoted in the text can be checked in one place.

## Design notes

| Item | How the code handles it |
|---|---|
| Observed exposure | Releases 4–5 only, because the work-use classification and task success are published from Release 4. Computed with the capability gate (β ≥ 0.5) and both with and without the work-usage gate (WorkUsage ≥ 100); files carry `gate_on`. |
| Concentration | `concentration` (raw conversation volume N) is the measure used in the thesis. `concentration_gated` (N restricted to gate-passing tasks) is also written to the construct files as a sensitivity check. |
| Effective exposure | Task coverage and effective exposure are weighted by time fraction (Section 3.2.3). |
| Entire-process marker | Figures 7, 13, 14 and 17 draw the entire process as a square at its process-level values: Rₒ,ₚ, process concentration, and the time-weighted expertise or value of all in-scope tasks. Subprocess bubbles are sized by conversation volume. |
| Section 4.2.2 values | `text_values_4_2_2.csv` gives the expertise at which the entire process, process concentration and observed exposure sit. |
| Table 6 use-case rows | Shares are conversation-pooled (summed conversation counts per category over in-scope tasks), so work, personal, coursework and not classified sum to 100%. `table06_use_case_diagnostic.csv` sets these beside the unweighted task mean for comparison. |
| Figure 26 | The partial regression controls for subprocess, Claude.ai collaboration mode and ridge expertise (`COMPLEXITY_CONTROL = "ridge"`). A sensitivity version controlling for AEI `human_education_years` is written as `fig26b_alt_aei_education_control_R5.png`. |
| Regressions | Tables 7–8 use conventional OLS standard errors without a release indicator. Table D8 re-estimates them with task-clustered (CR1) standard errors, with and without a Release 5 indicator. Tables D1 and D10 estimate them by release. |
| Release 6 | Not used in any construct. The panel is built from Releases 1–5, so the coverage checks in `0_constructs/` cover Releases 1–5. Adding the Release 6 raw files to `../data_preparation/inputs/aei/release_6/` and re-running that folder's notebook 04 adds it to the checks. |

## Status

Run end to end on 16 October 2026 (Python 3.12, Windows), on inputs built from the raw data by `../data_preparation/`, with no errors. The values reported in the thesis match this run.

## References

The constructs follow the thesis (Section 3.2). Full references and data licences are in the [root README](../../README.md#references).

- **Usage, collaboration modes, economic primitives, task success:** Anthropic Economic Index Releases 1–5 (Anthropic, 2026a; Handa et al., 2025; Appel et al., 2025, 2026; Massenkoff et al., 2026).
- **Capability gate (β ≥ 0.5):** Eloundou et al. (2024).
- **Time-fraction weights:** elicited following Tamkin & McCrory (2025), see `../data_preparation/01`.
- **Expertise axis:** ridge task expertise (`../data_preparation/02`), motivated by Autor & Thompson (2025).
- **Task value:** OEWS wages (U.S. Bureau of Labor Statistics, 2026b).
- **Table D6:** Krippendorff's α (Hayes & Krippendorff, 2007).
