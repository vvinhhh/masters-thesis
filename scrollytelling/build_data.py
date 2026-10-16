"""Build the data for the scrollytelling page and inline it into index.html.

Reads the same inputs and constructs as code/final_pipeline (via r2r_common), so every
number on the page matches the thesis. Run from anywhere:

    python scrollytelling/build_data.py

Writes scrollytelling/data.json and scrollytelling/index.html (template.html with the
data inlined, so the page works as a single file, even opened from disk), and copies the
page plus social-card.png into docs/, the folder GitHub Pages publishes.

social-card.png (the 1200x627 link-preview image) is drawn by social-card.html; regenerate it
by opening that page through a local server and saving the canvas.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "code" / "final_pipeline"
sys.path.insert(0, str(PIPELINE))
import r2r_common as rc  # noqa: E402

SHORT = {
    "Master Data Management": "Master Data",
    "Record": "Record",
    "Close & Consolidation": "Close & Consolidation",
    "Report": "Report",
    "Analysis & Business Insight": "Analysis & Insight",
    "Planning & Budgeting": "Planning & Budgeting",
}
R = 5  # the story's release


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def main():
    tm, panel = rc.load_data(verbose=False)
    tm = tm.set_index("task_id")
    tf = tm["task_time_fraction"].astype(float)
    w = tf / tf.sum()

    sub_exp = rc.read_construct("subprocess_exposure.csv")
    proc_exp = rc.read_construct("process_exposure.csv")
    eff = rc.read_construct("effective_exposure.csv")
    task_exp = rc.read_construct("task_exposure.csv").set_index(["task_id", "release", "gate_on"])
    idx = rc.read_construct("r2r_indices.csv")
    collab = pd.read_csv(rc.OUT_DIR / rc.OUT_SECTIONS["rq1"] / "fig02_04_data.csv")
    gini = pd.read_csv(rc.OUT_DIR / rc.OUT_SECTIONS["rq1"] / "fig10_gini_top20.csv")

    # ---- tasks ---------------------------------------------------------------
    success = rc.task_success_table(panel, R)
    hot = rc.get_task_metric(panel, R, "claude_ai", "human_only_time")
    autonomy = {p: rc.get_task_metric(panel, R, p, "ai_autonomy") for p in rc.PLATFORMS}
    api_idx = idx[(idx["release"] == R)].set_index("task_id")
    conv = {p: rc.task_series(panel, R, p, "usage", "count", None, "sum") for p in rc.PLATFORMS}
    touched = rc.get_covered_task_ids(panel, R, rc.RELEASE_PLATFORMS[R])

    by_release = {}
    for rel in rc.ANALYSIS_RELEASES:
        for p in rc.RELEASE_PLATFORMS[rel]:
            share = rc.get_r2r_task_pct(panel, rel, p)
            cs = rc.get_collaboration_shares(panel, rel, p, share.index) if rel >= 2 else None
            for t in share.index:
                d = by_release.setdefault(t, {}).setdefault(f"{p}_{rel}", {})
                d["u"] = r(share[t] * 100, 3)
                if cs is not None and t in cs.index:
                    d["a"] = r(cs.loc[t, "automation_share"] * 100, 1)
                    d["g"] = r(cs.loc[t, "augmentation_share"] * 100, 1)

    tasks = []
    for t, row in tm.iterrows():
        te = task_exp.loc[(t, R, True)]
        inp = api_idx["input_index"].get(t, np.nan)
        tasks.append({
            "id": t,
            "s": row["task_statement"],
            "sp": row["subprocess"],
            "occ": row["occupation_title"],
            "h": r(row["task_time_fraction"], 2),
            "w": r(w[t], 5),
            "yrs": r(row["ridge_task_years"], 2),
            "wage": r(row["wage_hourly_mean"], 2),
            "beta": r(row["beta"], 1),
            "touched": t in touched,
            "conv": {"claude_ai": r(conv["claude_ai"].get(t, 0), 0), "1p_api": r(conv["1p_api"].get(t, 0), 0)},
            "auto": r(te["AutoShare"] * 100, 1) if te["automation_count"] + te["augmentation_count"] > 0 else None,
            "work": r(te["WorkUsage"], 0),
            "rt": r(te["r_tilde"], 3),
            "gate": bool(te["gate_pass"]),
            "succ": {"claude_ai": r(success["claude_ai"].get(t), 1), "1p_api": r(success["1p_api"].get(t), 1)},
            "hot": r(hot.get(t), 2),
            "lnin": r(np.log(inp), 3) if inp and inp > 0 else None,
            "aut": {p: r(autonomy[p].get(t), 2) for p in rc.PLATFORMS},
            "rel": by_release.get(t, {}),
        })

    # ---- subprocess and process aggregates ------------------------------------
    subs = []
    for sp in rc.SUBPROCESS_ORDER:
        ids = tm.index[tm["subprocess"] == sp]
        ws = tf[ids] / tf[ids].sum()
        so = sub_exp[(sub_exp.subprocess == sp) & (sub_exp.release == R)].set_index("gate_on")
        e = eff[(eff.subprocess == sp) & (eff.release == R)].iloc[0]
        subs.append({
            "name": sp, "short": SHORT[sp], "n": int(len(ids)),
            "time_share": r(tf[ids].sum() / tf.sum(), 4),
            "cap": r(so.loc[True, "theoretical_capability_share"]),
            "obs_on": r(so.loc[True, "R_o"]), "obs_off": r(so.loc[False, "R_o"]),
            "eff": r(e["effective_exposure_pct"] / 100),
            "cov": r(e["task_coverage_pct"] / 100), "n_touched": int(e["n_touched"]),
            "touched_time": r(ws[[i in touched for i in ids]].sum()),
        })

    pe = proc_exp[(proc_exp.release == R)].set_index("gate_on")
    pe_eff = eff[(eff.subprocess == "Entire Process") & (eff.release == R)].iloc[0]
    capable = tm["beta"] >= rc.BETA_CAPABLE
    process = {
        "n": int(len(tm)),
        "n_capable": int(capable.sum()),
        "n_touched": int(pe_eff["n_touched"]),
        "cap_time": r(w[capable].sum()),
        "touched_time": r(w[[t in touched for t in tm.index]].sum()),
        "eff_time": r(pe_eff["effective_exposure_pct"] / 100),
        "obs_on": r(pe.loc[True, "R_process"]), "obs_off": r(pe.loc[False, "R_process"]),
    }

    c5 = collab[(collab.release == R) & (collab.subprocess == "Entire Process")].set_index("platform")
    channel = {p: {"auto": r(c5.loc[p, "automation_pct"], 1), "aug": r(c5.loc[p, "augmentation_pct"], 1)}
               for p in rc.PLATFORMS}
    g5 = gini[gini.release == R].set_index("platform")
    conc = {p: {"gini": r(g5.loc[p, "gini"], 3), "top20": r(g5.loc[p, "top20_share_pct"], 1)} for p in rc.PLATFORMS}

    fits = pd.read_csv(rc.OUT_DIR / rc.OUT_SECTIONS["rq2"] / "fit_statistics.csv")

    def fit(fig, panel_label):
        f = fits[(fits.figure == fig) & (fits.panel == panel_label)]
        f = f.iloc[0] if len(f) else None
        if f is None:
            return None
        return {"slope": r(f.slope, 3), "icpt": r(f.intercept, 3), "r2": r(f.r2, 3), "n": int(f.n)}

    # The Figure 21 and 22 fits are logged per panel; pick the rows the story uses.
    f21 = fits[(fits.figure == "Figure 21") & (fits.panel == "R5 Claude.ai")].iloc[0]
    f21a = fits[(fits.figure == "Figure 21") & (fits.panel == "R5 1P API")].iloc[0]
    f22 = fits[(fits.figure == "Figure 22") & (fits.panel == "R5 Claude.ai") & (fits.x == "Human-only Time (hours)")].iloc[0]
    drivers = {
        "expertise": {"slope": r(f21.slope, 3), "icpt": r(f21.intercept, 3), "r2": r(f21.r2, 3), "n": int(f21.n)},
        "expertise_api": {"slope": r(f21a.slope, 3), "icpt": r(f21a.intercept, 3), "r2": r(f21a.r2, 3), "n": int(f21a.n)},
        "hot": {"slope": r(f22.slope, 3), "icpt": r(f22.intercept, 3), "r2": r(f22.r2, 3), "n": int(f22.n)},
        "input": fit("Figure 24", "R5 ln_input"),
    }

    data = {"release": R, "subprocesses": subs, "process": process, "channel": channel,
            "concentration": conc, "drivers": drivers, "tasks": tasks}

    (HERE / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    html = (HERE / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = html.replace("/*__DATA__*/null", payload)
    # artifact.html: page body only (the claude.ai Artifact host adds the document skeleton).
    # index.html: a complete standalone document, for opening from disk or GitHub Pages. The title,
    # meta tags (incl. the LinkedIn/Open Graph preview) and styles go in <head>, where link crawlers look.
    (HERE / "artifact.html").write_text(page, encoding="utf-8")
    split = page.index('<header class="masthead">')
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  + page[:split] + '</head>\n<body>\n' + page[split:] + '\n</body>\n</html>\n')
    (HERE / "index.html").write_text(standalone, encoding="utf-8")
    # docs/: the public site, served by GitHub Pages from the main branch's /docs folder
    docs = HERE.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(standalone, encoding="utf-8")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    card = HERE / "social-card.png"
    if card.exists():
        (docs / "social-card.png").write_bytes(card.read_bytes())
    print(f"[write] data.json ({len(tasks)} tasks), artifact.html, index.html, docs/")
    print("process:", process)
    print("channel:", channel, "concentration:", conc)
    print("drivers:", drivers)


if __name__ == "__main__":
    main()
