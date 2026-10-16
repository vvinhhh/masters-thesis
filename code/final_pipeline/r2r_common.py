"""
r2r_common.py -- shared configuration, data loading, construct builders, regression
helpers and chart house style for the Record-to-Report (R2R) thesis pipeline.

Every notebook in this folder imports from here, so each construct (observed exposure,
concentration, effective exposure, R2R-relative indices, ...) is defined exactly ONCE.

Inputs (only two files, both in DATA_DIR):
    task_master.csv      one row per O*NET task (subprocess, beta, time fraction,
                         ridge expertise, wage, ...)
    final_aei_panel.csv  long-format AEI panel (release, platform, task_id, dimension,
                         category, level, variable, value)

DATA_DIR defaults to ./data next to this file; override with the environment variable
R2R_DATA_DIR (or pass data_dir= to load_data()). Outputs go to ./outputs, organised by
where they appear in the thesis (see OUT_SECTIONS below and README.md).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from scipy import stats as sp_stats

# =============================================================================
# 1. Paths and output layout
# =============================================================================
PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("R2R_DATA_DIR", PIPELINE_DIR / "data"))
OUT_DIR = Path(os.environ.get("R2R_OUT_DIR", PIPELINE_DIR / "outputs"))

TASK_MASTER_FILE = "task_master.csv"
PANEL_FILE = "final_aei_panel.csv"

# Output folders are named after WHERE the item sits in the thesis, so a figure can be
# found from its number alone. "constructs" holds intermediate CSVs that later notebooks
# read back in (01 writes them, 02-04 consume them).
OUT_SECTIONS = {
    "constructs": "0_constructs",
    "ch3": "1_ch3_descriptives",
    "rq1": "2_ch4_1_rq1_location",
    "rq2": "3_ch4_2_rq2_expertise",
    "appD": "4_appendix_D",
}


def out_path(section: str, filename: str) -> Path:
    """Full path for an output file, creating the section folder if needed."""
    folder = OUT_DIR / OUT_SECTIONS[section]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def save_csv(df: pd.DataFrame, section: str, filename: str, index: bool = False) -> None:
    path = out_path(section, filename)
    df.to_csv(path, index=index)
    print(f"[write] {path.relative_to(OUT_DIR.parent).as_posix()}")


def read_construct(filename: str, **kwargs) -> pd.DataFrame:
    """Read an intermediate table written by 01_constructs_and_checks.ipynb."""
    path = OUT_DIR / OUT_SECTIONS["constructs"] / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 01_constructs_and_checks.ipynb first.")
    df = pd.read_csv(path, dtype={"task_id": str}, **kwargs)
    if "gate_on" in df.columns:
        df["gate_on"] = _to_bool(df["gate_on"])
    return df


# =============================================================================
# 2. Analysis constants
# =============================================================================
SUBPROCESS_ORDER = [
    "Master Data Management",
    "Record",
    "Close & Consolidation",
    "Report",
    "Analysis & Business Insight",
    "Planning & Budgeting",
]
PLATFORMS = ["claude_ai", "1p_api"]
PLATFORM_LABELS = {"1p_api": "1P API", "claude_ai": "Claude.ai"}
RELEASE_LABELS = {i: f"Release {i}" for i in range(1, 7)}

# Which platforms publish R2R data in which release (1P API only from Release 3).
RELEASE_PLATFORMS = {
    1: ["claude_ai"], 2: ["claude_ai"],
    3: ["claude_ai", "1p_api"], 4: ["claude_ai", "1p_api"],
    5: ["claude_ai", "1p_api"], 6: ["claude_ai", "1p_api"],
}
ANALYSIS_RELEASES = (1, 2, 3, 4, 5)   # Release 6 is excluded throughout (Appendix B.2)
EXPOSURE_RELEASES = (4, 5)            # work-use classification + task success: R4-R5 only
GATE_STATES = (True, False)

AUTOMATION_CATEGORIES = {"directive", "feedback_loop"}
AUGMENTATION_CATEGORIES = {"learning", "task_iteration", "validation"}
NUMERIC_CATEGORY_PLACEHOLDERS = {"", "value"}   # R4 stores numeric facets blank, R5 as "value"

WORK_USAGE_GATE = 100        # volume gate: WorkUsage_t >= 100
BETA_CAPABLE = 0.5           # capability gate: beta_t >= 0.5


# =============================================================================
# 3. Data loading and panel accessors
# =============================================================================
def _norm_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in {"nan", "none", "na", "null", "n/a"}:
        return ""
    return s


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False}).astype(bool)


def load_data(data_dir=None, verbose: bool = True):
    """Load task_master.csv (in-scope tasks only) and final_aei_panel.csv.

    Adds normalised helper columns to the panel (platform_n, dimension_n, variable_n,
    category_n: lower-case, stripped, spaces -> underscores, so "feedback loop" and
    "feedback_loop" match). Release 6 rows are kept in the panel (for data checks) but
    no analysis function uses them.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    tm_path, panel_path = data_dir / TASK_MASTER_FILE, data_dir / PANEL_FILE
    missing = [p.name for p in (tm_path, panel_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Copy task_master.csv and final_aei_panel.csv "
            f"there, or set the R2R_DATA_DIR environment variable before importing r2r_common.")

    tm_all = pd.read_csv(tm_path, dtype={"task_id": str})
    tm_all["task_id"] = tm_all["task_id"].astype(str).str.strip()
    tm_all["subprocess"] = tm_all["subprocess"].astype(str).str.strip()
    dup = tm_all["task_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"task_master.csv has {dup.sum()} duplicated task_id rows.")
    required = ["subprocess", "beta", "task_time_fraction", "ridge_task_years", "wage_hourly_mean"]
    absent = [c for c in required if c not in tm_all.columns]
    if absent:
        raise ValueError(f"task_master.csv is missing required columns: {absent}")

    tm = tm_all[tm_all["subprocess"] != "OUT_OF_SCOPE"].copy().reset_index(drop=True)
    unknown = set(tm["subprocess"]) - set(SUBPROCESS_ORDER)
    if unknown:
        raise ValueError(f"Unknown subprocess labels in task_master.csv: {unknown}")
    bad_beta = ~tm["beta"].isin([0, 0.5, 1])
    if bad_beta.any():
        raise ValueError(f"{bad_beta.sum()} in-scope tasks have beta outside {{0, 0.5, 1}}.")

    panel = pd.read_csv(panel_path, dtype={"task_id": str}, low_memory=False)
    panel["task_id"] = panel["task_id"].astype(str).str.strip()
    rel = pd.to_numeric(panel["release"], errors="coerce")
    if rel.isna().mean() > 0.5:   # e.g. "release_4" -> 4
        rel = pd.to_numeric(panel["release"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    panel = panel.loc[rel.notna()].copy()
    panel["release"] = rel.loc[rel.notna()].astype(int)
    for col in ["platform", "dimension", "variable"]:
        panel[f"{col}_n"] = panel[col].map(_norm_str) if col in panel.columns else ""
    panel["category_n"] = (panel["category"].map(_norm_str).str.replace(" ", "_", regex=False)
                           if "category" in panel.columns else "")
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")

    if verbose:
        n_outside = (~panel["task_id"].isin(set(tm["task_id"]))).sum()
        print(f"[load] task_master: {len(tm_all)} rows, {len(tm)} in scope "
              f"({tm['subprocess'].value_counts().reindex(SUBPROCESS_ORDER).to_dict()})")
        print(f"[load] panel: {len(panel):,} rows, releases {sorted(panel['release'].unique())}, "
              f"{panel['task_id'].nunique()} tasks; rows whose task is not in scope: {n_outside}")
    return tm, panel


def task_series(panel, release, platform, dimension, variable, category=None, agg="mean") -> pd.Series:
    """task_id -> value for one panel slice.

    platform: "claude_ai", "1p_api", a list of both, or None (all platforms).
    category=None selects numeric/blank-category rows ("" or "value"); otherwise an
    exact (normalised) category such as "yes", "work", "directive".
    agg: how to combine several rows per task ("mean" for metrics, "sum" for counts).
    """
    m = ((panel["release"] == release)
         & (panel["dimension_n"] == dimension)
         & (panel["variable_n"] == variable))
    if platform is not None:
        plats = [platform] if isinstance(platform, str) else list(platform)
        m &= panel["platform_n"].isin(plats)
    if category is None:
        m &= panel["category_n"].isin(NUMERIC_CATEGORY_PLACEHOLDERS)
    else:
        m &= panel["category_n"] == category
    return panel.loc[m].groupby("task_id")["value"].agg(agg)


def get_task_metric(panel, release, platform, dimension, category=None, variable="mean") -> pd.Series:
    """Task-level metric (mean of the matching rows), e.g. ai_autonomy mean or task_success yes pct."""
    return task_series(panel, release, platform, dimension, variable, category=category, agg="mean")


def get_r2r_task_pct(panel, release, platform=None) -> pd.Series:
    """Task share of R2R usage (0-1), renormalised so the in-scope tasks of this
    release/platform slice sum to 1 (raw AEI pct is relative to the whole economy)."""
    m = (panel["release"] == release) & (panel["dimension_n"] == "usage") & (panel["variable_n"] == "pct")
    if platform is not None:
        plats = [platform] if isinstance(platform, str) else list(platform)
        m &= panel["platform_n"].isin(plats)
    grouped = panel.loc[m].groupby("task_id")["value"].sum()
    total = grouped.sum()
    return grouped if total == 0 else grouped / total


def get_covered_task_ids(panel, release, platform) -> set:
    """Tasks with ANY usage row on the platform(s) in this release ("touched" / "covered")."""
    plats = [platform] if isinstance(platform, str) else list(platform)
    m = (panel["release"] == release) & (panel["dimension_n"] == "usage") & panel["platform_n"].isin(plats)
    return set(panel.loc[m, "task_id"].unique())


def get_ridge_task_years(tm) -> pd.Series:
    return tm.set_index("task_id")["ridge_task_years"]


def get_collaboration_shares(panel, release, platform, task_ids=None) -> pd.DataFrame:
    """Per task: automation_share / augmentation_share (0-1 fractions of that task's own
    conversations), from the collaboration pct rows of ONE platform. Unclassified
    categories (none / not_classified / filtered) are dropped from both numerators, so
    the two shares need not sum to 1."""
    m = (panel["release"] == release) & (panel["dimension_n"] == "collaboration") & (panel["platform_n"] == platform)
    df = panel.loc[m, ["task_id", "category_n", "variable_n", "value"]]
    if task_ids is not None:
        df = df[df["task_id"].isin(set(task_ids))]
    if df.empty:
        return pd.DataFrame(columns=["automation_share", "augmentation_share"], dtype=float)
    if (df["variable_n"] == "pct").any():
        df = df[df["variable_n"] == "pct"]
    mode = np.select([df["category_n"].isin(AUTOMATION_CATEGORIES), df["category_n"].isin(AUGMENTATION_CATEGORIES)],
                     ["automation", "augmentation"], default="excluded")
    df = df.assign(mode=mode)
    df = df[df["mode"] != "excluded"]
    piv = df.groupby(["task_id", "mode"])["value"].sum().unstack(fill_value=0.0)
    piv = piv.rename(columns={"automation": "automation_share", "augmentation": "augmentation_share"})
    for col in ["automation_share", "augmentation_share"]:
        if col not in piv.columns:
            piv[col] = 0.0
    return piv[["automation_share", "augmentation_share"]]


def _weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    values, weights = values[mask], weights[mask]
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan
    return float((values * weights).sum() / weights.sum())


def scope_task_ids(tm, scope) -> list:
    """Task ids of a subprocess, or of the whole process when scope is None / "Entire Process"."""
    if scope is None or scope == "Entire Process":
        return tm["task_id"].tolist()
    return tm.loc[tm["subprocess"] == scope, "task_id"].tolist()


# =============================================================================
# 4. Collaboration-mode constructs (Section 4.1.1)
# =============================================================================
def compute_collab_mode_by_subprocess(tm, panel, platform, releases=(2, 3, 4, 5)) -> pd.DataFrame:
    """Usage-weighted automation / augmentation share (%) for the entire process and each
    subprocess. Each task's own collaboration shares are weighted by its R2R usage share
    on the same platform (Figures 2-4)."""
    rows = []
    sp_map = tm.set_index("task_id")["subprocess"]
    for r in releases:
        if platform not in RELEASE_PLATFORMS[r]:
            continue
        task_pct = get_r2r_task_pct(panel, r, platform)
        if task_pct.empty:
            continue
        collab = get_collaboration_shares(panel, r, platform, task_pct.index).reindex(task_pct.index).fillna(0.0)
        rows.append({"release": r, "platform": platform, "subprocess": "Entire Process",
                     "automation_pct": float((task_pct * collab["automation_share"]).sum()) * 100,
                     "augmentation_pct": float((task_pct * collab["augmentation_share"]).sum()) * 100})
        sp_of_task = np.asarray(task_pct.index.map(sp_map))
        for sp in SUBPROCESS_ORDER:
            mask = sp_of_task == sp
            w = task_pct[mask]
            if mask.sum() == 0 or w.sum() <= 0:
                continue
            w_norm = w / w.sum()
            ids = task_pct.index[mask]
            rows.append({"release": r, "platform": platform, "subprocess": sp,
                         "automation_pct": float((w_norm * collab.loc[ids, "automation_share"]).sum()) * 100,
                         "augmentation_pct": float((w_norm * collab.loc[ids, "augmentation_share"]).sum()) * 100})
    return pd.DataFrame(rows)


def compute_automation_overrepresentation(tm, panel, platform, releases=(2, 3, 4, 5)) -> pd.DataFrame:
    """Subprocess automation share / entire-process automation share, same platform and release (Figure 5)."""
    df = compute_collab_mode_by_subprocess(tm, panel, platform, releases)
    if df.empty:
        return df
    proc = df[df["subprocess"] == "Entire Process"].set_index("release")["automation_pct"]
    sub = df[df["subprocess"] != "Entire Process"].copy()
    sub["automation_index"] = sub["automation_pct"] / sub["release"].map(proc)
    return sub


# =============================================================================
# 5. Observed exposure (Section 3.2.1)
# =============================================================================
def _collab_bucket_counts(panel, release, bucket) -> pd.Series:
    """Claude.ai collaboration conversation COUNTS summed over the categories in `bucket`."""
    m = ((panel["release"] == release) & (panel["platform_n"] == "claude_ai")
         & (panel["dimension_n"] == "collaboration") & (panel["variable_n"] == "count")
         & panel["category_n"].isin(bucket))
    return panel.loc[m].groupby("task_id")["value"].sum()


def compute_task_exposure(tm, panel, release, gate_on=True) -> pd.DataFrame:
    """Task-level observed exposure r_tilde_t for one release and gate setting.

    WorkUsage_t = Claude.ai work-use conversations + all 1P API conversations
    AutoShare_t = automation / (automation + augmentation) Claude.ai conversations
    alpha_t     = 0.5 + 0.5 * (ClaudeWork_t * AutoShare_t + API_t) / WorkUsage_t
    r_tilde_t   = alpha_t if beta_t >= 0.5 and (gate off or WorkUsage_t >= 100), else 0
    """
    if release not in EXPOSURE_RELEASES:
        raise ValueError(f"Observed exposure needs the work-use classification (Releases {EXPOSURE_RELEASES}).")
    df = tm[["task_id", "subprocess", "beta", "task_time_fraction"]].copy().set_index("task_id")
    df["ClaudeWorkUsage"] = task_series(panel, release, "claude_ai", "use_case", "count", "work", "sum").reindex(df.index).fillna(0.0)
    df["APIUsage"] = task_series(panel, release, "1p_api", "usage", "count", None, "sum").reindex(df.index).fillna(0.0)
    df["automation_count"] = _collab_bucket_counts(panel, release, AUTOMATION_CATEGORIES).reindex(df.index).fillna(0.0)
    df["augmentation_count"] = _collab_bucket_counts(panel, release, AUGMENTATION_CATEGORIES).reindex(df.index).fillna(0.0)
    df["WorkUsage"] = df["ClaudeWorkUsage"] + df["APIUsage"]

    collab_total = df["automation_count"] + df["augmentation_count"]
    df["AutoShare"] = (df["automation_count"] / collab_total.where(collab_total > 0)).fillna(0.0)
    has_usage = df["WorkUsage"] > 0
    df["alpha"] = np.where(has_usage,
                           0.5 + 0.5 * (df["ClaudeWorkUsage"] * df["AutoShare"] + df["APIUsage"])
                           / df["WorkUsage"].where(has_usage), np.nan)
    df["gate_pass"] = (df["WorkUsage"] >= WORK_USAGE_GATE) if gate_on else True
    df["theoretical_capable"] = df["beta"] >= BETA_CAPABLE
    df["r_tilde"] = np.where(has_usage & df["gate_pass"] & df["theoretical_capable"], df["alpha"], 0.0)
    df["release"] = release
    df["gate_on"] = gate_on
    return df.reset_index()


def build_exposure_tables(tm, panel, releases=EXPOSURE_RELEASES, gates=GATE_STATES):
    """Returns (task_exposure, subprocess_exposure, process_exposure) for every release x gate."""
    task_level = pd.concat([compute_task_exposure(tm, panel, r, g) for r in releases for g in gates],
                           ignore_index=True)
    rows, prows = [], []
    for (r, g), grp in task_level.groupby(["release", "gate_on"]):
        prows.append({"release": r, "gate_on": g, "n_tasks": len(grp),
                      "n_tasks_with_usage": int((grp["WorkUsage"] > 0).sum()),
                      "n_tasks_gate_passed": int((grp["gate_pass"] & (grp["WorkUsage"] > 0)).sum()),
                      "theoretical_capability_share": grp["theoretical_capable"].mean(),
                      "R_process": _weighted_mean(grp["r_tilde"], grp["task_time_fraction"]),
                      "conversation_volume": grp["WorkUsage"].sum()})
        for sp, g2 in grp.groupby("subprocess"):
            rows.append({"subprocess": sp, "release": r, "gate_on": g, "n_tasks": len(g2),
                         "n_tasks_with_usage": int((g2["WorkUsage"] > 0).sum()),
                         "n_tasks_gate_passed": int((g2["gate_pass"] & (g2["WorkUsage"] > 0)).sum()),
                         "theoretical_capability_share": g2["theoretical_capable"].mean(),
                         "R_o": _weighted_mean(g2["r_tilde"], g2["task_time_fraction"]),
                         "conversation_volume": g2["WorkUsage"].sum()})
    return task_level, pd.DataFrame(rows), pd.DataFrame(prows)


# =============================================================================
# 6. Concentration (Section 3.2.2)
# =============================================================================
def compute_task_dispersion(task_level, sub_exp, proc_exp) -> pd.DataFrame:
    """Each task's time-weighted squared deviation from its subprocess mean (R_o) and from
    the pooled process mean (R_process); weights renormalised within the scope; zero-
    exposure tasks are kept."""
    df = task_level.copy()
    key_sub = ["subprocess", "release", "gate_on"]
    sub_r = sub_exp.set_index(key_sub)["R_o"]
    df["subprocess_R_o"] = df.set_index(key_sub).index.map(sub_r)
    proc_r = proc_exp.set_index(["release", "gate_on"])["R_process"]
    df["process_R"] = df.set_index(["release", "gate_on"]).index.map(proc_r)

    w = df["task_time_fraction"]
    sub_w_sum = df.groupby(key_sub)["task_time_fraction"].transform("sum")
    df["subprocess_weight_norm"] = np.where(sub_w_sum > 0, w / sub_w_sum, np.nan)
    df["subprocess_sq_dev"] = (df["r_tilde"] - df["subprocess_R_o"]) ** 2
    df["subprocess_weighted_dispersion"] = df["subprocess_weight_norm"] * df["subprocess_sq_dev"]

    proc_w_sum = df.groupby(["release", "gate_on"])["task_time_fraction"].transform("sum")
    df["process_weight_norm"] = np.where(proc_w_sum > 0, w / proc_w_sum, np.nan)
    df["process_sq_dev"] = (df["r_tilde"] - df["process_R"]) ** 2
    df["process_weighted_dispersion"] = df["process_weight_norm"] * df["process_sq_dev"]

    keep = ["task_id", "subprocess", "release", "gate_on", "task_time_fraction", "r_tilde", "WorkUsage", "gate_pass",
            "subprocess_R_o", "subprocess_weight_norm", "subprocess_sq_dev", "subprocess_weighted_dispersion",
            "process_R", "process_weight_norm", "process_sq_dev", "process_weighted_dispersion"]
    out = df[keep].sort_values(["release", "gate_on", "subprocess", "task_id"]).reset_index(drop=True)
    for (r, g), grp in out.groupby(["release", "gate_on"]):
        assert np.isclose(grp["process_weight_norm"].sum(), 1.0), "process weights must sum to 1"
    return out


def _concentration_record(grp, dispersion_col):
    raw = grp[dispersion_col].sum()
    n_volume = float(grp["WorkUsage"].sum())
    n_volume_gated = float(grp.loc[grp["gate_pass"].astype(bool), "WorkUsage"].sum())
    return {"n_tasks": len(grp), "conversation_volume": n_volume, "conversation_volume_gated": n_volume_gated,
            "raw_dispersion_sum": raw,
            "concentration": raw * np.log1p(n_volume),               # used in the thesis
            "concentration_gated": raw * np.log1p(n_volume_gated)}   # sensitivity variant


def compute_concentration(task_disp):
    """Returns (subprocess_concentration, process_concentration). Concentration =
    sum_t w_t (r_t - R)^2 x ln(1 + N), N = raw conversation volume (gate-invariant)."""
    sub_rows = []
    for (r, g, sp), grp in task_disp.groupby(["release", "gate_on", "subprocess"]):
        rec = {"subprocess": sp, "release": r, "gate_on": g, "R_o": grp["subprocess_R_o"].iloc[0]}
        rec.update(_concentration_record(grp, "subprocess_weighted_dispersion"))
        sub_rows.append(rec)
    proc_rows = []
    for (r, g), grp in task_disp.groupby(["release", "gate_on"]):
        rec = {"release": r, "gate_on": g, "R_process": grp["process_R"].iloc[0]}
        rec.update(_concentration_record(grp, "process_weighted_dispersion"))
        proc_rows.append(rec)
    return pd.DataFrame(sub_rows), pd.DataFrame(proc_rows)


# =============================================================================
# 7. Task success and effective exposure (Section 3.2.3)
# =============================================================================
def task_success_table(panel, release) -> pd.DataFrame:
    """Per task: success (%) on each platform and the pooled mean across platforms with a reading."""
    cols = {p: get_task_metric(panel, release, p, "task_success", category="yes", variable="pct") * 100
            for p in RELEASE_PLATFORMS[release]}
    df = pd.DataFrame(cols)
    df["success_pooled"] = df.mean(axis=1, skipna=True)
    df.index.name = "task_id"
    return df


def compute_effective_exposure_table(tm, panel, releases=EXPOSURE_RELEASES) -> pd.DataFrame:
    """Effective exposure = sum_t w_t * success_t over ALL in-scope tasks of the scope
    (untouched tasks contribute 0), w_t = time fraction renormalised within the scope.
    Task coverage = share of in-scope tasks touched on either platform (count basis)."""
    tf = tm.set_index("task_id")["task_time_fraction"].astype(float)
    rows = []
    for r in releases:
        touched = get_covered_task_ids(panel, r, RELEASE_PLATFORMS[r])
        success = task_success_table(panel, r)["success_pooled"].reindex(tf.index).fillna(0.0)
        for scope in ["Entire Process"] + SUBPROCESS_ORDER:
            ids = scope_task_ids(tm, scope)
            w = tf.loc[ids] / tf.loc[ids].sum()
            n_touched = sum(i in touched for i in ids)
            rows.append({"release": r, "subprocess": scope, "n_in_scope": len(ids), "n_touched": n_touched,
                         "task_coverage_pct": 100.0 * n_touched / len(ids),
                         "effective_exposure_pct": float((w * success.loc[ids]).sum())})
    return pd.DataFrame(rows)


def decompose_coverage_gap(tm, panel, releases=EXPOSURE_RELEASES) -> pd.DataFrame:
    """Exact decomposition of (effective exposure - task coverage), in pp, per scope:

        E - C = 100 * sum_touched (w_t - 1/N) * s_t      <- weight component
              + 100 * sum_touched (s_t - 1) / N           <- success component
              + 100 * sum_untouched w_t * s_t             <- residual (should be 0)

    with s_t = success_t / 100 and N = tasks in scope. A negative weight component means
    the touched tasks carry less of the scope's time than their count share (low-weight
    tasks); the success component (<= 0) is the shortfall from tasks Claude touches but
    does not reliably complete (low-success tasks)."""
    tf = tm.set_index("task_id")["task_time_fraction"].astype(float)
    rows = []
    for r in releases:
        touched = get_covered_task_ids(panel, r, RELEASE_PLATFORMS[r])
        s_all = task_success_table(panel, r)["success_pooled"].reindex(tf.index).fillna(0.0) / 100.0
        for scope in ["Entire Process"] + SUBPROCESS_ORDER:
            ids = scope_task_ids(tm, scope)
            n = len(ids)
            w = tf.loc[ids] / tf.loc[ids].sum()
            s = s_all.loc[ids]
            t = pd.Series([i in touched for i in ids], index=ids)
            weight_c = 100 * float(((w - 1 / n) * s)[t].sum())
            success_c = 100 * float(((s - 1) / n)[t].sum())
            resid = 100 * float((w * s)[~t].sum())
            eff, cov = 100 * float((w * s).sum()), 100.0 * t.sum() / n
            rows.append({"release": r, "subprocess": scope, "n_in_scope": n, "n_touched": int(t.sum()),
                         "task_coverage_pct": cov, "effective_exposure_pct": eff, "gap_pp": eff - cov,
                         "weight_component_pp": weight_c, "success_component_pp": success_c,
                         "residual_pp": resid,
                         "touched_low_weight_n": int(((w < 1 / n) & t).sum()),
                         "touched_low_success_n": int(((s < 0.5) & t).sum()),
                         "touched_low_weight_and_low_success_n": int(((w < 1 / n) & (s < 0.5) & t).sum()),
                         "mean_success_touched_pct": 100 * float(s[t].mean()) if t.any() else np.nan,
                         "time_share_touched_pct": 100 * float(w[t].sum())})
    out = pd.DataFrame(rows)
    assert np.allclose(out["gap_pp"], out["weight_component_pp"] + out["success_component_pp"] + out["residual_pp"])
    return out


# =============================================================================
# 8. Expertise coverage gap (Section 4.2.1) and R2R-relative indices (Section 4.2.4)
# =============================================================================
def compute_net_skill_coverage_gap(tm, panel, releases=ANALYSIS_RELEASES) -> pd.DataFrame:
    """Time-weighted mean ridge expertise of all in-scope tasks minus that of covered tasks."""
    t = tm.set_index("task_id")
    ridge, w = t["ridge_task_years"], t["task_time_fraction"]
    rows = []
    for r in releases:
        for platform in RELEASE_PLATFORMS[r]:
            covered = get_covered_task_ids(panel, r, platform) & set(t.index)
            for scope in ["Entire Process"] + SUBPROCESS_ORDER:
                ids = scope_task_ids(tm, scope)
                cov_ids = [i for i in ids if i in covered]
                mean_all = _weighted_mean(ridge.reindex(ids), w.reindex(ids))
                mean_cov = _weighted_mean(ridge.reindex(cov_ids), w.reindex(cov_ids))
                rows.append({"release": r, "platform": platform, "subprocess": scope, "n_covered": len(cov_ids),
                             "mean_all": mean_all, "mean_covered": mean_cov, "gap": mean_all - mean_cov})
    return pd.DataFrame(rows)


def compute_r2r_index(panel, tm, release, dimension, platform="1p_api") -> pd.Series:
    """Economy-wide AEI index (cost / prompt_tokens / completion_tokens) re-indexed to the
    R2R task set: divide by the unweighted mean over in-scope tasks with a value, so the
    R2R average is 1.0."""
    raw = get_task_metric(panel, release, platform, dimension, variable="index")
    raw = raw.reindex(tm["task_id"]).dropna()
    return raw if raw.empty else raw / raw.mean()


def build_r2r_indices(panel, tm, releases=EXPOSURE_RELEASES) -> pd.DataFrame:
    """Long table: task_id, release, cost_index, input_index, output_index (R2R-relative)."""
    frames = []
    for r in releases:
        d = pd.DataFrame({"cost_index": compute_r2r_index(panel, tm, r, "cost"),
                          "input_index": compute_r2r_index(panel, tm, r, "prompt_tokens"),
                          "output_index": compute_r2r_index(panel, tm, r, "completion_tokens")})
        d.index.name = "task_id"
        d = d.reset_index()
        d["release"] = r
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["subprocess"] = out["task_id"].map(tm.set_index("task_id")["subprocess"])
    return out


# =============================================================================
# 9. Regression helpers (Tables 7, 8, D1, D8)
# =============================================================================
def ols_fit(X, y, clusters=None):
    """OLS. Conventional (homoskedastic) SEs by default -- the thesis tables use these.
    With `clusters` (array of group ids) returns CR1 cluster-robust SEs, p-values on G-1 df."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    if clusters is None:
        dof = max(n - k, 1)
        var = (float(resid @ resid) / dof) * xtx_inv
    else:
        clusters = np.asarray(clusters)
        groups = pd.unique(clusters)
        g = len(groups)
        meat = np.zeros((k, k))
        for grp in groups:
            idx = clusters == grp
            s = X[idx].T @ resid[idx]
            meat += np.outer(s, s)
        scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
        var = scale * xtx_inv @ meat @ xtx_inv
        dof = max(g - 1, 1)
    se = np.sqrt(np.clip(np.diag(var), 0, None))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * sp_stats.t.sf(np.abs(t), df=dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return dict(beta=beta, se=se, t=t, p=p, r2=r2, n=n, k=k, dof=dof, resid=resid)


def design(df, cols, intercept=True):
    X = df[cols].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(df)), X]) if intercept else X


def add_subprocess_dummies(df, ref=SUBPROCESS_ORDER[0]):
    """Subprocess fixed-effect dummies (reference: Master Data Management)."""
    df = df.copy()
    cols = []
    for sp in SUBPROCESS_ORDER:
        if sp == ref:
            continue
        col = "sp_" + sp.replace(" & ", "_").replace(" ", "_")
        df[col] = (df["subprocess"] == sp).astype(float)
        cols.append(col)
    return df, cols


def joint_f_test(df, y_col, restricted_cols, full_cols):
    fr = ols_fit(design(df, restricted_cols), df[y_col])
    ff = ols_fit(design(df, full_cols), df[y_col])
    ssr_r, ssr_f = float(fr["resid"] @ fr["resid"]), float(ff["resid"] @ ff["resid"])
    q, dof = ff["k"] - fr["k"], ff["dof"]
    if q <= 0 or ssr_f <= 0:
        return dict(F=np.nan, p=np.nan, q=q, dof=dof)
    F = ((ssr_r - ssr_f) / q) / (ssr_f / dof)
    return dict(F=float(F), p=float(sp_stats.f.sf(F, q, dof)), q=q, dof=dof)


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def run_stepwise(df, y_col, specs, cluster_col=None, fe_test=None):
    """Fit each (label, cols) spec on the rows complete for THAT spec. Returns
    {label: {"coefs": {name: {coef, se, p}}, "r2", "n"}} plus "_ftest" if fe_test=(restricted, full)."""
    out = {}
    for label, cols in specs:
        sub = df.dropna(subset=cols + [y_col])
        fit = ols_fit(design(sub, cols), sub[y_col], clusters=sub[cluster_col].values if cluster_col else None)
        names = ["intercept"] + cols
        out[label] = {"coefs": {nm: dict(coef=fit["beta"][i], se=fit["se"][i], p=fit["p"][i])
                                for i, nm in enumerate(names)},
                      "r2": fit["r2"], "n": fit["n"]}
    if fe_test is not None:
        restricted, full = fe_test
        out["_ftest"] = joint_f_test(df.dropna(subset=full + [y_col]), y_col, restricted, full)
    return out


def regression_table(results, row_order, fe_label="Subprocess fixed effects", fe_from=None) -> pd.DataFrame:
    """Thesis-style table: coefficient row (with stars) and SE row beneath, then FE indicator,
    F-test, observations, R^2 -- the same layout as Tables 7 and 8."""
    ftest = results.get("_ftest")
    specs = [k for k in results if not k.startswith("_")]
    rows = []
    for key, label in row_order:
        coef_row, se_row = [label], [""]
        for s in specs:
            c = results[s]["coefs"].get(key)
            coef_row.append("" if c is None else f"{c['coef']:.3f}{stars(c['p'])}")
            se_row.append("" if c is None else f"({c['se']:.3f})")
        rows += [coef_row, se_row]
    if fe_from is not None:
        rows.append([fe_label] + ["Yes" if i >= fe_from else "No" for i in range(len(specs))])
        if ftest is not None and not np.isnan(ftest["F"]):
            rows.append(["F-test of fixed effects"] + ["" if i != len(specs) - 1 else
                        f"F({ftest['q']}, {ftest['dof']}) = {ftest['F']:.2f} [p = {ftest['p']:.3f}]"
                        for i in range(len(specs))])
    rows.append(["Observations"] + [str(results[s]["n"]) for s in specs])
    rows.append(["R²"] + [f"{results[s]['r2']:.3f}" for s in specs])
    return pd.DataFrame(rows, columns=[""] + specs)


def build_claude_success_dataset(tm, panel, releases=EXPOSURE_RELEASES, platform="claude_ai") -> pd.DataFrame:
    """Task x release rows for Table 7: ridge expertise, AI autonomy, human-only time (hours),
    task success (%), subprocess. Unit = task x release (releases pooled when several)."""
    ridge = get_ridge_task_years(tm)
    sp_map = tm.set_index("task_id")["subprocess"]
    frames = []
    for r in releases:
        autonomy = get_task_metric(panel, r, platform, "ai_autonomy")
        idx = autonomy.index
        frames.append(pd.DataFrame({
            "task_id": idx, "release": r,
            "expertise": ridge.reindex(idx).values,
            "autonomy": autonomy.values,
            "human_only_time": get_task_metric(panel, r, platform, "human_only_time").reindex(idx).values,
            "task_success_pct": (get_task_metric(panel, r, platform, "task_success", "yes", "pct") * 100).reindex(idx).values,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["subprocess"] = df["task_id"].map(sp_map)
    return df.dropna(subset=["expertise", "autonomy", "human_only_time", "task_success_pct", "subprocess"]).reset_index(drop=True)


def build_api_success_dataset(tm, panel, releases=EXPOSURE_RELEASES, platform="1p_api") -> pd.DataFrame:
    """Task x release rows for Table 8: ln R2R-relative input/output indices, AI autonomy,
    task success (%), subprocess (+ ridge expertise for the correlation table only)."""
    ridge = get_ridge_task_years(tm)
    sp_map = tm.set_index("task_id")["subprocess"]
    frames = []
    for r in releases:
        inp = compute_r2r_index(panel, tm, r, "prompt_tokens")
        outp = compute_r2r_index(panel, tm, r, "completion_tokens")
        ids = inp.index
        with np.errstate(divide="ignore"):
            frames.append(pd.DataFrame({
                "task_id": ids, "release": r,
                "ln_input_index": np.log(inp.values),
                "ln_output_index": np.log(outp.reindex(ids).values),
                "autonomy": get_task_metric(panel, r, platform, "ai_autonomy").reindex(ids).values,
                "task_success_pct": (get_task_metric(panel, r, platform, "task_success", "yes", "pct") * 100).reindex(ids).values,
                "expertise": ridge.reindex(ids).values,
            }))
    df = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    df["subprocess"] = df["task_id"].map(sp_map)
    return df.dropna(subset=["ln_input_index", "ln_output_index", "autonomy", "task_success_pct", "subprocess"]).reset_index(drop=True)


# =============================================================================
# 10. Fit-statistics log (every slope / R^2 the text quotes ends up in one CSV)
# =============================================================================
_FIT_LOG: list = []


def fit_line(x, y):
    """OLS line through finite (x, y) pairs -> (slope, intercept, r2, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    slope, intercept = np.polyfit(x, y, 1)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2), int(len(x))


def record_fit(figure, panel_label, x_name, y_name, slope, intercept, r2, n, note=""):
    _FIT_LOG.append(dict(figure=figure, panel=panel_label, x=x_name, y=y_name, slope=slope,
                         intercept=intercept, r2=r2, n=n, note=note))


def write_fit_log(section, filename="fit_statistics.csv", reset=True):
    if not _FIT_LOG:
        return None
    df = pd.DataFrame(_FIT_LOG)
    path = save_csv(df, section, filename)
    if reset:
        _FIT_LOG.clear()
    return df


# =============================================================================
# 11. Chart house style
# =============================================================================
SHOW_FIGURES = True   # set False to only write PNGs (faster, no inline output)

SUBPROCESS_COLORS = {
    "Master Data Management": "#6FA8A9", "Record": "#84A889", "Close & Consolidation": "#E8BE72",
    "Report": "#E39F5E", "Analysis & Business Insight": "#D17B73", "Planning & Budgeting": "#9884BE",
}
SUBPROCESS_LABELS_WRAPPED = {
    "Master Data Management": "Master Data\nManagement", "Record": "Record",
    "Close & Consolidation": "Close &\nConsolidation", "Report": "Report",
    "Analysis & Business Insight": "Analysis &\nBusiness Insight", "Planning & Budgeting": "Planning &\nBudgeting",
}
PLATFORM_COLORS = {"1p_api": "#4B90DD", "claude_ai": "#F88F49"}
PLATFORM_MARKERS = {"1p_api": "s", "claude_ai": "o"}
COLLAB_MODE_COLORS = {"automation": "#4B90DD", "augmentation": "#F88F49"}
COLLAB_MODE_MARKERS = {"automation": "s", "augmentation": "o"}
ENTIRE_PROCESS_COLOR = "#4B90DD"
ENTIRE_PROCESS_MARKER = "s"
ENTIRE_PROCESS_SIZE = 420

TEXT_BLACK = "#000000"
NEUTRAL_COLOR = "#D97757"
GRID_COLOR = "#D8D8D3"
GRID_LINEWIDTH = 0.5
BASELINE_COLOR = "#8C8B85"
BASELINE_LINEWIDTH = 1.0
QUADRANT_COLOR = "#A18877"
LIGHT_GREY = "#9C9B95"
MODE_COLOR_COOL = "#4D4C48"
AXIS_LINE_COLOR = "#000000"
LEGEND_EDGE_COLOR = "#C9C9C4"
LEGEND_FACE_COLOR = "#FFFFFF"
TEXT_HALO = [pe.withStroke(linewidth=1.5, foreground="white")]
AXIS_CLIP_PAD_FRAC = 0.03
TABLE_ROUNDING = 0.012

_FONT_FALLBACKS = ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]


def register_fonts():
    for font_dir in (PIPELINE_DIR / "fonts", Path.cwd() / "fonts"):
        if font_dir.is_dir():
            for f in font_dir.glob("*.ttf"):
                try:
                    fm.fontManager.addfont(str(f))
                except Exception:
                    pass
    available = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in _FONT_FALLBACKS if f in available), "sans-serif")


def apply_style():
    plt.rcParams.update({
        "font.family": register_fonts(), "font.size": 11, "text.color": TEXT_BLACK,
        "axes.edgecolor": AXIS_LINE_COLOR, "axes.linewidth": 1.1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.labelcolor": TEXT_BLACK, "axes.labelweight": "bold", "axes.labelsize": 12, "axes.labelpad": 14,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "xtick.labelsize": 10, "ytick.labelsize": 10, "xtick.color": TEXT_BLACK, "ytick.color": TEXT_BLACK,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 10,
    })


apply_style()


def save_figure(fig, section, filename, show=None) -> None:
    path = out_path(section, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[write] {path.relative_to(OUT_DIR.parent).as_posix()}")
    # show inline in Jupyter; skip under a non-interactive backend (e.g. MPLBACKEND=Agg for headless runs)
    if (show if show is not None else SHOW_FIGURES) and "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def clean_axes(ax, grid="both"):
    ax.grid(False)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, linestyle="-", color=GRID_COLOR, linewidth=GRID_LINEWIDTH, zorder=0)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, linestyle="-", color=GRID_COLOR, linewidth=GRID_LINEWIDTH, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS_LINE_COLOR)
    ax.spines["bottom"].set_color(AXIS_LINE_COLOR)


def capped_ylim(ax, hard_cap, pad_frac=AXIS_CLIP_PAD_FRAC, bottom=0.0):
    ax.set_ylim(bottom, hard_cap * (1 + pad_frac))


def release_ticks(ax, releases, short=False):
    releases = list(releases)
    ax.set_xticks(releases)
    ax.set_xticklabels([f"R{r}" for r in releases] if short else [RELEASE_LABELS[r] for r in releases])


def baseline_hline(ax, y=1.0):
    ax.axhline(y, color=BASELINE_COLOR, linewidth=BASELINE_LINEWIDTH, linestyle="--", zorder=1.5)


def wrap_label(text, max_len=18):
    if len(text) <= max_len:
        return text
    words = text.split(" ")
    mid = len(words) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def legend_box(target, entries, loc="upper right", ncol=None, fontsize=9.5, bbox_to_anchor=None, title=None):
    """Bordered legend; entries = [(color, label, marker)] with marker "line", "s" or a marker code."""
    if ncol is None:
        ncol = 2 if len(entries) > 3 else 1
    handles = []
    for color, label, marker in entries:
        if marker == "line":
            h = plt.Line2D([0], [0], color=color, linestyle="--", linewidth=1.6)
        elif marker in (None, "s", "square"):
            h = plt.Line2D([0], [0], marker="s", linestyle="None", color=color, markersize=9, markeredgecolor="none")
        else:
            h = plt.Line2D([0], [0], marker=marker, linestyle="None", color=color, markersize=8,
                           markeredgecolor="white", markeredgewidth=0.6)
        h.set_label(label)
        handles.append(h)
    leg = target.legend(handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, frameon=True,
                        fancybox=True, edgecolor=LEGEND_EDGE_COLOR, facecolor=LEGEND_FACE_COLOR, framealpha=0.97,
                        fontsize=fontsize, labelcolor=TEXT_BLACK, handletextpad=0.6, columnspacing=1.2,
                        borderpad=0.7, title=title)
    leg.get_frame().set_linewidth(1.0)
    return leg


def stagger_labels(items, min_gap, y_min=None, y_max=None):
    ordered = sorted(items, key=lambda t: t[0])
    adjusted = [v for v, _ in ordered]
    for i in range(1, len(adjusted)):
        if adjusted[i] - adjusted[i - 1] < min_gap:
            adjusted[i] = adjusted[i - 1] + min_gap
    if y_max is not None and adjusted and adjusted[-1] > y_max:
        adjusted = [v - (adjusted[-1] - y_max) for v in adjusted]
    if y_min is not None and adjusted and adjusted[0] < y_min:
        adjusted = [v + (y_min - adjusted[0]) for v in adjusted]
    return {ordered[i][1]: adjusted[i] for i in range(len(ordered))}


def end_of_line_labels(ax, last_vals, colors, x_pos, min_gap_frac=0.055, min_gap_abs=0.12,
                       fontsize=10, fontweight="semibold", clamp=False, dx=10):
    if not last_vals:
        return
    y_span = max(last_vals.values()) - min(last_vals.values())
    min_gap = max(y_span * min_gap_frac, min_gap_abs)
    ylim = ax.get_ylim() if clamp else (None, None)
    label_y = stagger_labels([(v, k) for k, v in last_vals.items()], min_gap, ylim[0], ylim[1])
    for key, y in label_y.items():
        ax.annotate(key, (x_pos, y), xytext=(dx, 0), textcoords="offset points", va="center", ha="left",
                    color=colors[key], fontsize=fontsize, fontweight=fontweight, zorder=6,
                    path_effects=TEXT_HALO, clip_on=False)


def bubble_sizes(volumes, min_size=50, max_size=1400, vmax=None):
    v = pd.Series(volumes).clip(lower=0).to_numpy(dtype=float)
    vmax = v.max() if vmax is None else vmax
    if vmax <= 0:
        return np.full_like(v, min_size)
    return min_size + (max_size - min_size) * np.sqrt(v / vmax)


def global_limits(dfs, x_col, y_col, xpad_frac=0.12, ypad_frac=0.15, y_floor_zero=True):
    """One (xlim, ylim) spanning all frames, so the panels of a figure share a scale."""
    dfs = [d for d in dfs if len(d)]
    all_x = pd.concat([d[x_col] for d in dfs], ignore_index=True).dropna()
    all_y = pd.concat([d[y_col] for d in dfs], ignore_index=True).dropna()
    xpad = (all_x.max() - all_x.min()) * xpad_frac or 1.0
    ypad = (all_y.max() - all_y.min()) * ypad_frac or 1.0
    y_low = min(0, all_y.min() - ypad) if y_floor_zero else all_y.min() - ypad
    return (all_x.min() - xpad, all_x.max() + xpad), (y_low, all_y.max() + ypad)


def add_quadrant_lines(ax, xlim, ylim, x_mid=None, y_mid=None):
    ax.axvline((xlim[0] + xlim[1]) / 2 if x_mid is None else x_mid, color=QUADRANT_COLOR, linestyle="--",
               linewidth=BASELINE_LINEWIDTH, zorder=1)
    ax.axhline((ylim[0] + ylim[1]) / 2 if y_mid is None else y_mid, color=QUADRANT_COLOR, linestyle="--",
               linewidth=BASELINE_LINEWIDTH, zorder=1)


def trend_line(ax, x, y, xlim, color=BASELINE_COLOR):
    """Draw the OLS line across xlim; returns (slope, intercept, r2, n, legend_label)."""
    slope, intercept, r2, n = fit_line(x, y)
    if n >= 2:
        xl = np.linspace(xlim[0], xlim[1], 100)
        ax.plot(xl, slope * xl + intercept, color=color, linestyle="--", linewidth=1.5, zorder=4)
    sign = "+" if intercept >= 0 else "-"
    return slope, intercept, r2, n, f"Fit: y = {slope:.3f}x {sign} {abs(intercept):.3f}\n$R^2$ = {r2:.3f}"


def draw_entire_process(ax, x, y, label="Entire Process"):
    """The blue square marking the entire process on subprocess-level scatters (Figures 7, 13, 14, 17)."""
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return
    ax.scatter([x], [y], s=ENTIRE_PROCESS_SIZE, color=ENTIRE_PROCESS_COLOR, marker=ENTIRE_PROCESS_MARKER,
               alpha=0.85, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate(label, (x, y), xytext=(0, 14), textcoords="offset points", ha="center", va="bottom",
                color=ENTIRE_PROCESS_COLOR, fontsize=8, fontweight="semibold", zorder=6, path_effects=TEXT_HALO)


def subprocess_bubbles(ax, d, x_col, y_col, size_col="conversation_volume", fontsize=7):
    """One labelled bubble per subprocess, area-proportional to size_col."""
    sizes = bubble_sizes(d[size_col]) if size_col in d.columns else np.full(len(d), 300.0)
    ax.scatter(d[x_col], d[y_col], s=sizes, color=d["subprocess"].map(SUBPROCESS_COLORS), alpha=0.82,
               edgecolor="white", linewidth=1.0, zorder=3)
    for (_, row), sz in zip(d.iterrows(), sizes):
        ax.annotate(wrap_label(row["subprocess"], 20), (row[x_col], row[y_col]),
                    xytext=(0, 6 + 0.02 * np.sqrt(sz)), textcoords="offset points", ha="center", va="bottom",
                    color=SUBPROCESS_COLORS.get(row["subprocess"], TEXT_BLACK), fontsize=fontsize,
                    fontweight="semibold", zorder=6, path_effects=TEXT_HALO)


def task_bubbles(ax, d, x_col, y_col, size_col="WorkUsage", top_n=6, label_col="task_id"):
    """One bubble per task coloured by subprocess; the top_n highest-y tasks are labelled."""
    ax.scatter(d[x_col], d[y_col], s=bubble_sizes(d[size_col]), color=d["subprocess"].map(SUBPROCESS_COLORS),
               alpha=0.65, edgecolor="white", linewidth=0.6, zorder=3)
    for _, row in d.nlargest(top_n, y_col).iterrows():
        ax.annotate(str(row[label_col]), (row[x_col], row[y_col]), xytext=(7, 5), textcoords="offset points",
                    ha="left", va="bottom", color=SUBPROCESS_COLORS.get(row["subprocess"], TEXT_BLACK),
                    fontsize=6, zorder=6, path_effects=TEXT_HALO)


def scatter_by_subprocess(ax, df, xcol, ycol, s=42, alpha=0.78):
    for sp in SUBPROCESS_ORDER:
        sub = df[df["subprocess"] == sp]
        if not sub.empty:
            ax.scatter(sub[xcol], sub[ycol], color=SUBPROCESS_COLORS[sp], s=s, alpha=alpha,
                       edgecolor="white", linewidth=0.4, zorder=3, label=sp)


def binned_scatter_with_trend(ax, x, y, color, marker, n_bins=15, s=70, label=None):
    """Binned means (quantile bins of x) as markers + OLS line fitted on the RAW task points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x_c, y_c = x[m], y[m]
    slope, intercept, r2, n = fit_line(x_c, y_c)
    if n < 2:
        return r2, n, slope, intercept
    q = max(2, min(n_bins, len(np.unique(x_c))))
    try:
        bins = pd.qcut(x_c, q=q, duplicates="drop")
        b = pd.DataFrame({"x": x_c, "y": y_c, "bin": bins}).groupby("bin", observed=True).agg(
            x_mean=("x", "mean"), y_mean=("y", "mean")).dropna()
        bx, by = b["x_mean"].values, b["y_mean"].values
    except (ValueError, IndexError):
        bx, by = x_c, y_c
    ax.scatter(bx, by, color=color, marker=marker, s=s, zorder=3, edgecolor="white", linewidth=0.8, label=label)
    xl = np.linspace(x_c.min(), x_c.max(), 60)
    ax.plot(xl, slope * xl + intercept, color=color, linestyle="--", linewidth=1.7, zorder=2)
    return r2, n, slope, intercept


def platform_stats_box(ax, stats_by_platform, anchor=(0.03, 0.97), fontsize=9.5, line_gap=0.025, box_width=0.28):
    order = [p for p in PLATFORMS if p in stats_by_platform]
    if not order:
        return
    x0, y0 = anchor
    pad = 0.015
    box_h = (len(order) * 2 - 1) * line_gap + 2 * pad
    ax.add_patch(mpatches.FancyBboxPatch((x0, y0 - box_h), box_width, box_h,
                 boxstyle=f"round,pad=0.0,rounding_size={TABLE_ROUNDING}", transform=ax.transAxes,
                 facecolor=LEGEND_FACE_COLOR, edgecolor=LEGEND_EDGE_COLOR, linewidth=1.0, zorder=8, clip_on=False))
    line = 0.05
    for p in order:
        s = stats_by_platform[p]
        ax.text(x0 + pad, y0 - pad - line * line_gap, f"{PLATFORM_LABELS[p]}  (R² = {s['r2']:.3f}, n = {s['n']})",
                transform=ax.transAxes, color=PLATFORM_COLORS[p], fontsize=fontsize, fontweight="semibold",
                va="center", ha="left", zorder=9)
        line += 1
        ax.text(x0 + pad, y0 - pad - line * line_gap, f"y = {s['slope']:.3f}x + {s['intercept']:.3f}",
                transform=ax.transAxes, color=PLATFORM_COLORS[p], fontsize=fontsize - 1, fontstyle="italic",
                va="center", ha="left", zorder=9)
        line += 1
