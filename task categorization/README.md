# Phase 1 – Task categorisation into R2R subprocesses (Section 3.1.3, Appendix A, C.2–C.3)

This phase maps the 212 O\*NET task statements of the ten study occupations to the six Record-to-Report subprocesses or `OUT_OF_SCOPE`:

- Master Data Management
- Record
- Close & Consolidation
- Report
- Analysis & Business Insight
- Planning & Budgeting

Three LLMs (Claude Sonnet 4.6, GPT-5.4-mini, Gemini-3.1-flash-lite) and a human rater classify every task independently. Disputed tasks then go through a reasoning-sharing consensus round.

**Output:** [`final_task_register.csv`](final_task_register.csv), with one row per task: final label, resolution path, and the vote tallies before and after consensus. It is the fixed starting point of phase 2 (`code/data_preparation/inputs/final_task_register.csv`).

| Notebook / file | Step |
|---|---|
| `acc.thes task categorization.md` | method notes: rationale, codebook design, validity run, production run, consensus round |
| `1_classification.ipynb` | production classification prompt (subprocess definitions, decision rules, system prompt; Appendix C.2) and API calls |
| `2_agreement_script.ipynb` | production-run agreement: Krippendorff's α (nominal), pairwise and full agreement, confusion matrices, OUT_OF_SCOPE rates |
| `june 22.1 consensus.ipynb` | consensus round for the 81 non-unanimous tasks (prompt in Appendix C.3) and its vote tally |
| `final_register.ipynb` | merges unanimous, consensus and human-resolved labels into the final register |
| `final_task_register_summary.csv` | label counts by resolution path (Table A1) |

**Re-running.** These notebooks are kept as a record of the procedure. They made paid API calls and read the human rater's classification files, which are not part of the repository, so they are not meant to be re-run. Their saved outputs show the results reported in Appendix A. Everything downstream starts from `final_task_register.csv`.

## References

O\*NET task statements: National Center for O\*NET Development (2026), O\*NET 30.2 (CC BY 4.0; see the attribution in the [root README](../README.md#data-sources-and-licences)).

Method precedents and annotation literature cited in Appendix A:

- Autor, D., & Thompson, N. (2025). Expertise. *Journal of the European Economic Association, 23*(4), 1203–1271. https://doi.org/10.1093/jeea/jvaf023
- Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2024). GPTs are GPTs: Labor market impact potential of LLMs. *Science, 384*(6702), 1306–1308. https://doi.org/10.1126/science.adj0998
- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *Proceedings of the National Academy of Sciences, 120*(30), e2305016120. https://doi.org/10.1073/pnas.2305016120
- Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. *Communication Methods and Measures, 1*(1), 77–89. https://doi.org/10.1080/19312450709336664
- Törnberg, P. (2023). ChatGPT-4 outperforms experts and crowd workers in annotating political Twitter messages with zero-shot learning. *arXiv preprint* arXiv:2304.06588.
- Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., & Yang, D. (2024). Can large language models transform computational social science? *Computational Linguistics, 50*(1), 237–291. https://doi.org/10.1162/coli_a_00502
- Parshall and Lopez-Luzuriaga (2026): multi-model (Claude, GPT, Gemini) consensus classification of O\*NET tasks, the design this phase adapts.
