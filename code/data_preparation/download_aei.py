"""
download_aei.py -- fetch the raw Anthropic Economic Index (AEI) files used in this thesis.

The raw AEI files are too large for the repository (the Release 5 Claude.ai file is 103 MB),
so they are downloaded from Anthropic's public dataset on Hugging Face
(https://huggingface.co/datasets/Anthropic/EconomicIndex, data released under CC BY) into
inputs/aei/release_<n>/, where 04_aei_panel.ipynb expects them. Every file is checked
against the SHA-256 of the exact file used in the thesis.

Usage (from this folder):
    python download_aei.py                # Releases 1-5 (what the analysis uses, about 310 MB)
    python download_aei.py --release6     # also Release 6 (about 300 MB; excluded from the analysis)

Already-present files with the right checksum are skipped. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main"
AEI_DIR = Path(__file__).resolve().parent / "inputs" / "aei"

# release -> list of (path in the Hugging Face dataset, SHA-256 of the file used in the thesis)
FILES = {
    1: [("release_2025_02_10/onet_task_mappings.csv",
         "8de89c99f762146c9e46b2202c3501e2a14f11c5b44c61c8007021127b6c676e")],
    2: [("release_2025_03_27/task_pct_v2.csv",
         "0c20f184cd4a1e3d8ef3d77f500b32b62431ce61a3c2f133c3af7fa8429f97db"),
        ("release_2025_03_27/automation_vs_augmentation_by_task.csv",
         "f6570714a9af31ae783fd91bd7ee3b2e9cfabcbf522a30ba1aa9b1f230665e79")],
    3: [("release_2025_09_15/data/intermediate/aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv",
         "c8ef9c5eee0c42febc73e358ecc7d2358e0a0ce3b50122c0c15ae8ec569aceff"),
        ("release_2025_09_15/data/intermediate/aei_raw_1p_api_2025-08-04_to_2025-08-11.csv",
         "c0fc061ea7d4feb71a7bd88c528d54e2d3a11e2045574f1e1fb2d594eebffd0f")],
    4: [("release_2026_01_15/data/intermediate/aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv",
         "70dc16780061b84d3f34ddf7fe5c978d65c99a5ef58afd78f9315f886a6a5873"),
        ("release_2026_01_15/data/intermediate/aei_raw_1p_api_2025-11-13_to_2025-11-20.csv",
         "432f8d14348ba384dd9c659fa7fdc85754cbae3b5a480c14a344ac4823d0e861")],
    5: [("release_2026_03_24/data/aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv",
         "69ebf6f9afc1e3be45caab172a84ae09e78bf41b97e1c310100d81c8a35a7433"),
        ("release_2026_03_24/data/aei_raw_1p_api_2026-02-05_to_2026-02-12.csv",
         "b3bcd68e7f6d820ffbb50a3c53d71ec2a122556fc2a38226cd86951a054bd4c8")],
    # Release 6 was not used in the thesis, so there is no reference checksum.
    6: [("release_2026_06_26/data/aei_claude_ai_2026-06-26.csv", None),
        ("release_2026_06_26/data/aei_1p_api_2026-06-26.csv", None)],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(release: int, hf_path: str, expected: str | None) -> bool:
    dest = AEI_DIR / f"release_{release}" / Path(hf_path).name
    if dest.exists() and (expected is None or sha256(dest) == expected):
        print(f"  ok (already present)  {dest.relative_to(AEI_DIR)}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading           {dest.relative_to(AEI_DIR)} ...", flush=True)
    urllib.request.urlretrieve(f"{BASE_URL}/{hf_path}", tmp)
    if expected is not None and sha256(tmp) != expected:
        tmp.unlink()
        print(f"  CHECKSUM MISMATCH     {hf_path}: the published file differs from the one used in the thesis")
        return False
    tmp.replace(dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release6", action="store_true", help="also download Release 6 (not used in the analysis)")
    args = ap.parse_args()
    releases = [1, 2, 3, 4, 5] + ([6] if args.release6 else [])
    ok = True
    for r in releases:
        print(f"Release {r}")
        for hf_path, expected in FILES[r]:
            ok &= fetch(r, hf_path, expected)
    print("All files present and verified." if ok else "Some files failed verification (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
