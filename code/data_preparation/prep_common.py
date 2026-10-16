"""
prep_common.py -- shared paths, file names, ID handling and hand-off helpers for the
data-preparation phase of the Record-to-Report (R2R) thesis (Section 3.1).

Every notebook in this folder imports from here, so file locations and the task-id /
task-text normalisation rules are defined exactly ONCE.

Phase 1 (task categorisation, ../../task categorization/) ends with
    inputs/final_task_register.csv      212 O*NET tasks with their final R2R label

This phase turns that register into the two analysis inputs used by ../final_pipeline/:

    01_time_fraction_elicitation.ipynb   Section 3.1.1   -> elicited_time_fractions.csv
    02_ridge_task_expertise.ipynb        Section 3.1.2   -> ridge_task_complexity.csv
    03_task_master.ipynb                 Section 3.1.3   -> task_master.csv
    04_aei_panel.ipynb                   Section 3.1.4   -> final_aei_panel.csv

Raw inputs live in ./inputs and are never written to. Outputs go to ./outputs, one folder
per notebook (see OUT_SECTIONS). The files the analysis phase needs are also copied into
../final_pipeline/data/ by handoff(), so running 01-04 here and then 01-05 there is the
whole pipeline end to end.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pandas as pd

# =============================================================================
# 1. Paths and output layout
# =============================================================================
PREP_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path(os.environ.get("R2R_PREP_INPUT_DIR", PREP_DIR / "inputs"))
OUT_DIR = Path(os.environ.get("R2R_PREP_OUT_DIR", PREP_DIR / "outputs"))
# Where the analysis phase (../final_pipeline) reads its inputs from.
ANALYSIS_DATA_DIR = Path(os.environ.get("R2R_DATA_DIR", PREP_DIR.parent / "final_pipeline" / "data"))

# Raw inputs (all under INPUT_DIR)
REGISTER_FILE = INPUT_DIR / "final_task_register.csv"            # phase 1 output (Appendix A, Table A2)
ONET_TASKS_FILE = INPUT_DIR / "onet" / "task_statements.xlsx"    # O*NET 30.x Task Statements
ONET_RATINGS_FILE = INPUT_DIR / "onet" / "task_ratings.xlsx"     # O*NET Task Ratings (FT / IM / RT scales)
ONET_CATEGORIES_FILE = INPUT_DIR / "onet" / "task_categories.xlsx"  # labels of the FT frequency categories
BLS_EDUCATION_FILE = INPUT_DIR / "bls" / "BLS_education.xlsx"   # BLS Employment Projections Table 5.4
OEWS_WAGE_FILE = INPUT_DIR / "bls" / "OEWS_Wage_May2025.xlsx"   # BLS OEWS national estimates, May 2025
ELOUNDOU_FILE = INPUT_DIR / "eloundou" / "eloundou_beta.tsv"    # Eloundou et al. (2024) task-level exposure
TF_CACHE_FILE = INPUT_DIR / "time_fraction" / "elicitation_cache.json"  # raw model replies (Appendix C.1)
AEI_DIR = INPUT_DIR / "aei"                                      # release_1 ... release_6 sub-folders

# Output folders are named after the notebook (and thesis section) that writes them.
OUT_SECTIONS = {
    "time_fraction": "01_time_fraction",      # Section 3.1.1, Appendix C.1, Table D6
    "ridge": "02_ridge_expertise",            # Section 3.1.2, Table D5
    "task_master": "03_task_master",          # Section 3.1.3, Table B1
    "aei_panel": "04_aei_panel",              # Section 3.1.4, Appendix B
}


def out_path(section: str, filename: str) -> Path:
    """Full path for an output file, creating the section folder if needed."""
    folder = OUT_DIR / OUT_SECTIONS[section]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def show_path(path: Path) -> str:
    """Path relative to this folder, so printed messages never contain machine-specific paths."""
    try:
        return Path(os.path.relpath(path, PREP_DIR)).as_posix()
    except ValueError:            # different drive (Windows)
        return Path(path).name


def save_csv(df: pd.DataFrame, section: str, filename: str, index: bool = False) -> None:
    path = out_path(section, filename)
    df.to_csv(path, index=index, lineterminator="\n")   # same bytes on Windows, macOS and Linux
    print(f"[write] {show_path(path)}  ({len(df):,} rows)")


def save_text(text: str, section: str, filename: str) -> None:
    path = out_path(section, filename)
    path.write_text(text, encoding="utf-8")
    print(f"[write] {show_path(path)}")


def read_output(section: str, filename: str, **kwargs) -> pd.DataFrame:
    """Read a file written by a preceding notebook in this pipeline (task_id kept as string)."""
    path = OUT_DIR / OUT_SECTIONS[section] / filename
    if not path.exists():
        raise FileNotFoundError(f"{show_path(path)} not found -- run the notebook that writes "
                                f"outputs/{OUT_SECTIONS[section]}/ first (see README.md run order).")
    dtype = {"task_id": str, **kwargs.pop("dtype", {})}
    return pd.read_csv(path, dtype=dtype, **kwargs)


def handoff(section: str, filename: str) -> None:
    """Copy a finished output into ../final_pipeline/data/ for the analysis phase."""
    src = OUT_DIR / OUT_SECTIONS[section] / filename
    ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = ANALYSIS_DATA_DIR / filename
    shutil.copyfile(src, dst)
    print(f"[handoff] {show_path(src)} -> {show_path(dst)}")


def banner(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# 2. Identifiers and task text
# =============================================================================
# O*NET column names (identical in task_statements.xlsx, task_ratings.xlsx and the
# Eloundou file)
ONET_SOC_COL = "O*NET-SOC Code"
ONET_TITLE_COL = "Title"
ONET_TASKID_COL = "Task ID"
ONET_TASK_COL = "Task"
ONET_TASKTYPE_COL = "Task Type"

OUT_OF_SCOPE = "OUT_OF_SCOPE"


def normalize_id(series: pd.Series) -> pd.Series:
    """task_id as an opaque string: '1298', never 1298 or '1298.0'."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def normalize_task_text(s):
    """Normalise task text so the same task matches across AEI releases and O*NET
    (lower-cased, whitespace collapsed, trailing period stripped; Appendix B.1)."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".")


def soc6(onet_soc: pd.Series) -> pd.Series:
    """O*NET-SOC '11-3031.01' -> 6-digit SOC '11-3031' (the key used by BLS tables)."""
    return onet_soc.astype(str).str.split(".").str[0].str.strip()


# =============================================================================
# 3. Loaders for the shared raw inputs
# =============================================================================
def load_register() -> pd.DataFrame:
    """Final R2R task register from phase 1 (212 rows; final_label = subprocess or OUT_OF_SCOPE)."""
    df = pd.read_csv(REGISTER_FILE, dtype={"task_id": str})
    missing = {"task_id", "task_statement", "final_label"} - set(df.columns)
    if missing:
        raise ValueError(f"{REGISTER_FILE.name} is missing columns {missing}")
    df["task_id"] = normalize_id(df["task_id"])
    if df["task_id"].duplicated().any():
        raise ValueError("task_id is not unique in the register")
    return df


def load_onet_tasks() -> pd.DataFrame:
    """O*NET task statements (all occupations), task_id normalised to string."""
    df = pd.read_excel(ONET_TASKS_FILE)
    missing = {ONET_SOC_COL, ONET_TITLE_COL, ONET_TASKID_COL, ONET_TASK_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{ONET_TASKS_FILE.name} is missing columns {missing}")
    df[ONET_TASKID_COL] = normalize_id(df[ONET_TASKID_COL])
    return df


def in_scope_ids(register: pd.DataFrame) -> set:
    return set(register.loc[register["final_label"] != OUT_OF_SCOPE, "task_id"])
