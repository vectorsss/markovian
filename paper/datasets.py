"""Central registry of paper datasets and their source result directories.

Source directories under ``raw_results/`` are auto-detected: for each dataset
key, the newest matching subdir wins, and adjusted-R² runs are preferred
for regression datasets when available.

To pin a specific run instead of auto-detect, add an entry to
``SHAPG_OVERRIDES`` or ``MARKOVIAN_OVERRIDES`` below.
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RAW = _HERE / "raw_results"

# Output of the figure/table scripts (untracked), same layout as the paper repo:
# assets/figures/*.pdf and assets/table_*.tex.
ASSETS_DIR = _HERE / "assets"
FIGURES_DIR = ASSETS_DIR / "figures"

# Per-dataset metadata.  ``raw_name`` is the dataset token used inside the
# run-directory filenames (e.g. "crime" in INFO but "crimedata" on disk).
INFO = {
    "housing": {"label": "Housing", "n_features":  13, "is_regression": True,  "raw_name": "housing"},
    "h1n1":    {"label": "H1N1",    "n_features":  35, "is_regression": False, "raw_name": "h1n1"},
    "crime":   {"label": "Crime",   "n_features": 120, "is_regression": True,  "raw_name": "crimedata"},
    "mic":     {"label": "MIC",     "n_features": 111, "is_regression": False, "raw_name": "mic"},
}

# Dataset groupings — which datasets feed which comparison table.
SMALL = ["housing", "h1n1"]
BIG   = ["crime",   "mic"]
ALL   = SMALL + BIG

# Manual overrides — set ``{key: "subdir_basename"}`` to pin a specific run
# instead of auto-selecting the latest.  Leave empty for full auto.
SHAPG_OVERRIDES: dict = {}
MARKOVIAN_OVERRIDES: dict = {}


def _latest_dir(prefix: str, key: str):
    """Newest ``raw_results/`` subdir matching ``<prefix>_<rawname>_retrain*``.

    For regression datasets, runs whose name contains ``_adj_r2_`` are
    preferred; we only fall back to non-adj_r2 runs if no adj_r2 run exists.

    Returns the subdir *basename* or ``None`` if nothing matches.
    """
    raw_name = INFO[key]["raw_name"]
    pattern = f"{prefix}_{raw_name}_retrain*"
    if not _RAW.is_dir():
        return None
    matches = sorted(d.name for d in _RAW.glob(pattern) if d.is_dir())
    if not matches:
        return None

    if INFO[key]["is_regression"]:
        adj_only = [m for m in matches if "_adj_r2_" in m]
        if adj_only:
            matches = adj_only

    return matches[-1]  # alphabetical sort == timestamp sort for our naming


def _resolve(prefix: str, overrides: dict):
    """Build a key → subdir-basename map applying overrides + auto-detect."""
    return {k: (overrides.get(k) or _latest_dir(prefix, k)) for k in INFO}


# Auto-populated subdir maps.  Values may be ``None`` if no matching run
# exists yet; consumers raise a clear error at access time.
SHAPG_DIRS     = _resolve("shapg_comparison",   SHAPG_OVERRIDES)
MARKOVIAN_DIRS = _resolve("markovian_initial",  MARKOVIAN_OVERRIDES)


# ---- Path helpers ---------------------------------------------------------

def _check(name, family, key):
    if name is None:
        raise FileNotFoundError(
            f"No {family} run found for '{key}' under {_RAW}. "
            f"Either drop a matching run into raw_results/ or add a manual "
            f"entry in {family.upper()}_OVERRIDES (see datasets.py)."
        )
    return name


def shapg_path(key: str) -> str:
    """Path (relative to paper/) of the shapg_comparison run for *key*."""
    return f"raw_results/{_check(SHAPG_DIRS.get(key), 'shapg_comparison', key)}"


def markovian_path(key: str) -> str:
    """Path (relative to paper/) of the markovian_initial run for *key*."""
    return f"raw_results/{_check(MARKOVIAN_DIRS.get(key), 'markovian_initial', key)}"


# ---- LaTeX-label helpers --------------------------------------------------

def metric_label(key: str, dir_basename=None, full: bool = False) -> str:
    """LaTeX metric label.

    Auto-detects adjusted R² from ``_adj_r2`` in *dir_basename* when given.
    ``full=True`` appends a ``\\mathrm{full}`` subscript (used in the
    consolidated comparison tables).
    """
    is_reg = INFO[key]["is_regression"]
    adj = bool(dir_basename and "_adj_r2" in dir_basename)

    if not is_reg:
        return r"\mathrm{acc}_{\mathrm{full}}" if full else r"\mathrm{acc}"
    if adj and full:
        return r"R^2_{\mathrm{adj},\mathrm{full}}"
    if adj:
        return r"R^2_{\mathrm{adj}}"
    if full:
        return r"R^2_{\mathrm{full}}"
    return r"R^2"


def threshold_key(key: str) -> str:
    """Method key for the threshold-init Markovian variant (used in comparison rows)."""
    return "markovian_corr06" if INFO[key]["is_regression"] else "markovian_mi003"


# ---- Diagnostics ---------------------------------------------------------

def print_resolution():
    """Print which run each dataset key currently resolves to (debug helper)."""
    print(f"raw_results: {_RAW}")
    for family, dmap in (("shapg_comparison", SHAPG_DIRS),
                         ("markovian_initial", MARKOVIAN_DIRS)):
        print(f"\n[{family}]")
        for k, v in dmap.items():
            tag = ""
            if v is None:
                tag = "  (NO MATCH)"
            elif INFO[k]["is_regression"] and "_adj_r2_" not in v:
                tag = "  (no adj_r2 found, fell back)"
            print(f"  {k:8s} -> {v or '---'}{tag}")


if __name__ == "__main__":
    print_resolution()
