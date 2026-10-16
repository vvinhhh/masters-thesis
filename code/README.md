# R2R thesis – code map

The whole empirical pipeline is three phases. Each phase's output is the next phase's input.

| Phase | Folder | Thesis | Output |
|---|---|---|---|
| 1. Task categorisation | `../task categorization/` | Section 3.1.3, Appendix A, C.2–C.3 | `final_task_register.csv` (212 O\*NET tasks with their R2R subprocess or OUT_OF_SCOPE) |
| 2. Data preparation | [`data_preparation/`](data_preparation/README.md) | Section 3.1, Appendix B, C.1, Tables D5–D6 | `task_master.csv`, `final_aei_panel.csv`, `elicited_time_fractions.csv`, all copied into `final_pipeline/data/` |
| 3. Analysis | [`final_pipeline/`](final_pipeline/README.md) | Sections 3.2–3.3, Chapter 4, Appendix D | every figure and table in the thesis |

**To rebuild everything:** run `data_preparation` notebooks 01 → 04, then `final_pipeline` notebooks 01 → 05. Phase 1 involved LLM API calls and human review, so it is not re-run; its register is the fixed starting point.

## Where to find outputs (for charts, slides and write-ups)

Every table is a CSV. Every figure is a PNG next to the CSV of the data it plots.

| You want… | Look in |
|---|---|
| Per-task time fractions, per-model hours, model agreement | `data_preparation/outputs/01_time_fraction/` |
| Ridge task expertise, model diagnostics (Table D5) | `data_preparation/outputs/02_ridge_expertise/` |
| One row per task with every task-level variable | `data_preparation/outputs/03_task_master/task_master.csv` |
| The AEI panel, build log, release coverage (Table B3) | `data_preparation/outputs/04_aei_panel/` |
| Constructs per task / subprocess / release (exposure, concentration, success, indices) | `final_pipeline/outputs/0_constructs/` |
| Section 3.3 descriptives (Tables 2–3) | `final_pipeline/outputs/1_ch3_descriptives/` |
| Section 4.1 figures (1–10) | `final_pipeline/outputs/2_ch4_1_rq1_location/` |
| Section 4.2 figures and tables (11–26, Tables 4–8) | `final_pipeline/outputs/3_ch4_2_rq2_expertise/` |
| Appendix D | `final_pipeline/outputs/4_appendix_D/` |
