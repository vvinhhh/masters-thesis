# "You Will Be Replaced!" for the 1000th time: A Task-Level Analysis of Generative AI Exposure in the Record-to-Report Process

Replication code for the master's thesis by **Vinh Le** (Master's Programme in Accounting, 2026; supervisor Prof. Jukka Sihvonen).

The thesis measures *observed* generative-AI exposure in the Record-to-Report (R2R) accounting process at the task level, using five releases of the Anthropic Economic Index (AEI). It asks three questions:
- where within R2R exposure is concentrated;
- at what level of expertise it sits;
- how effective that exposure is once task success is taken into account.

## Repository layout

```
task categorization/        Phase 1  O*NET tasks -> R2R subprocesses (multi-LLM + human classification, consensus round)
code/data_preparation/      Phase 2  time fractions, ridge task expertise, task master, AEI panel   (thesis Section 3.1)
code/final_pipeline/        Phase 3  constructs, descriptives, RQ1, RQ2, Appendix D                (Sections 3.2-4.2)
requirements.txt
```

Each phase's output is the next phase's input. Every table is written as a CSV, and every figure as a PNG next to the CSV of the data it plots. [`code/README.md`](code/README.md) says where each kind of output lives. The READMEs in [`code/data_preparation/`](code/data_preparation/README.md) and [`code/final_pipeline/`](code/final_pipeline/README.md) document every notebook, input and output.

## Reproducing the results

```bash
pip install -r requirements.txt
```

```bash
python code/data_preparation/download_aei.py
```

The second command fetches the raw AEI files (about 310 MB) from Anthropic's public dataset and verifies them by checksum.

Then run, each from its own folder:

1. `code/data_preparation/` notebooks `01` → `04`. They hand their outputs to `code/final_pipeline/data/`.
2. `code/final_pipeline/` notebooks `01` → `05`.

For example, run headless from inside each folder:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_*.ipynb
```

Runtime is a few minutes, plus 10–20 minutes the first time `data_preparation/02` encodes the O\*NET task statements. **No API keys are needed.** The time-fraction model replies are cached in the repository, and phase 1's final register is the fixed starting point of phase 2. The full chain was verified end to end on 16 October 2026 (Python 3.12).

## Data sources and licences

| Data | Source | Licence | In this repository |
|---|---|---|---|
| Anthropic Economic Index, Releases 1–5 (Anthropic, 2026a; Handa et al., 2025; Appel et al., 2025, 2026; Massenkoff et al., 2026) | [huggingface.co/datasets/Anthropic/EconomicIndex](https://huggingface.co/datasets/Anthropic/EconomicIndex) | CC BY (data) | downloaded by `download_aei.py`; the derived panel `final_aei_panel.csv` is included |
| O\*NET 30.2 Task Statements and Task Ratings (National Center for O\*NET Development, 2026) | [onetcenter.org/database.html](https://www.onetcenter.org/database.html) | CC BY 4.0 | included (`code/data_preparation/inputs/onet/`) |
| BLS Employment Projections, education and training assignments by detailed occupation (U.S. Bureau of Labor Statistics, 2026a) | [bls.gov/emp/tables/education-and-training-by-occupation.htm](https://www.bls.gov/emp/tables/education-and-training-by-occupation.htm) | public domain (U.S. government work) | included (`inputs/bls/BLS_education.xlsx`) |
| BLS Occupational Employment and Wage Statistics, May 2025 national estimates (U.S. Bureau of Labor Statistics, 2026b) | [bls.gov/oes/tables.htm](https://www.bls.gov/oes/tables.htm) | public domain (U.S. government work) | included (`inputs/bls/OEWS_Wage_May2025.xlsx`) |
| Task-level GPT exposure ratings, β (Eloundou et al., 2024) | [github.com/openai/GPTs-are-GPTs](https://github.com/openai/GPTs-are-GPTs), `data/full_labelset.tsv` | MIT | included, unchanged (`inputs/eloundou/eloundou_beta.tsv`) |

O\*NET attribution: *This repository includes information from the O\*NET 30.2 Database by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA). Used under the CC BY 4.0 license. O\*NET® is a trademark of USDOL/ETA. The O\*NET information was modified by the author (filtered to the study occupations and combined with other sources); USDOL/ETA has not approved, endorsed, or tested these modifications.*

The AEI is built from Clio, Anthropic's privacy-preserving analysis pipeline (Tamkin et al., 2024; Anthropic, 2026b). This repository contains only Anthropic's published aggregates, restricted to R2R tasks, and nothing at a finer granularity.

## References

Thesis references used by the code:

- Anthropic. (2026a). *The Anthropic Economic Index*. https://www.anthropic.com/economic-index
- Anthropic. (2026b). *Enabling independent research on how people use Claude*. Anthropic. https://www.anthropic.com/research/enabling-independent-research
- Appel, R., Massenkoff, M., McCrory, P., McCain, M., Heller, R., Neylon, T., & Tamkin, A. (2026). *Anthropic Economic Index report: Economic primitives*. Anthropic. https://www.anthropic.com/research/anthropic-economic-index-january-2026-report
- Appel, R., McCrory, P., Tamkin, A., Stern, M., McCain, M., & Neylon, T. (2025). *Anthropic Economic Index report: Uneven geographic and enterprise AI adoption*. Anthropic. https://www.anthropic.com/research/anthropic-economic-index-september-2025-report
- Autor, D., & Thompson, N. (2025). Expertise. *Journal of the European Economic Association, 23*(4), 1203–1271. https://doi.org/10.1093/jeea/jvaf023
- Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2024). GPTs are GPTs: Labor market impact potential of LLMs. *Science, 384*(6702), 1306–1308. https://doi.org/10.1126/science.adj0998
- Handa, K., Tamkin, A., McCain, M., Huang, S., Durmus, E., Heck, S., Mueller, J., Hong, J., Ritchie, S., & Belonax, T. (2025). Which economic tasks are performed with AI? Evidence from millions of Claude conversations. *arXiv preprint* arXiv:2503.04761.
- Massenkoff, M., Lyubich, E., McCrory, P., Appel, R., & Heller, R. (2026). *Anthropic Economic Index report: Learning curves*. https://www.anthropic.com/research/economic-index-march-2026-report
- National Center for O\*NET Development. (2026). *O\*NET 30.2 data dictionary*. https://www.onetcenter.org/dictionary/30.2/excel/
- Tamkin, A., McCain, M., Handa, K., Durmus, E., Lovitt, L., Rathi, A., Huang, S., Mountfield, A., Hong, J., Ritchie, S., Stern, M., Clarke, B., Goldberg, L., Sumers, T. R., Mueller, J., McEachen, W., Mitchell, W., Carter, S., Clark, J., … Ganguli, D. (2024). Clio: Privacy-preserving insights into real-world AI use. *arXiv preprint* arXiv:2412.13678.
- Tamkin, A., & McCrory, P. (2025). *Estimating AI productivity gains from Claude conversations*. https://www.anthropic.com/research/estimating-productivity-gains
- U.S. Bureau of Labor Statistics. (2026a). *Education and training assignments by detailed occupation*. https://www.bls.gov/emp/tables/education-and-training-by-occupation.htm
- U.S. Bureau of Labor Statistics. (2026b). *Occupational Employment and Wage Statistics (OEWS) tables*. https://www.bls.gov/oes/tables.htm

Software and methods:

- Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. *Communication Methods and Measures, 1*(1), 77–89. https://doi.org/10.1080/19312450709336664 (computed with the `krippendorff` Python package)
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992. https://doi.org/10.18653/v1/D19-1410
- Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). MPNet: Masked and permuted pre-training for language understanding. *Advances in Neural Information Processing Systems, 33*, 16857–16867. (`all-mpnet-base-v2` sentence-embedding model)

## Citation

> Le, V. (2026). *"You Will Be Replaced!" for the 1000th time: A task-level analysis of generative AI exposure in the Record-to-Report process* [Master's thesis].
