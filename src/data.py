"""Data loading and feature assembly for the RadFusion reproduction.

Pipeline
--------
1. (optional) download the RadFusion CSVs into ./data
2. load the provided Vision.csv  -> study id, PE label, PENet probability, pe_type, split
3. build the structured EHR feature matrix by merging the EHR tables on `idx`
4. join the EHR matrix to the provided vision scores on `idx`

NOTE ON SCHEMA: RadFusion distributes most EHR features already encoded (one-hot / binary / counts),
each column keyed by `idx`. The steps below (rare-column removal, de-duplication, zero-variance
removal, standardization) follow the dataset's protocol. If your local copy has a different column
layout, adjust `build_ehr_features` accordingly — the join key `idx` and the Vision.csv columns are
the stable contract.
"""
from __future__ import annotations
import argparse
from functools import reduce
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------------------
def download(force: bool = False) -> None:
    """Fetch the RadFusion CSVs into ./data using gdown.

    Requires `gdown` and acceptance of the Stanford AIMI data-use agreement. The canonical source is
    https://aimi.stanford.edu/datasets ; the Drive IDs are the tutorial copies (see config).
    """
    import gdown  # local import so the rest of the package works without it

    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, file_id in C.GDRIVE_FILES.items():
        dest = C.DATA_DIR / name
        if dest.exists() and not force:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name}")
        gdown.download(id=file_id, output=str(dest), quiet=False)


# --------------------------------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------------------------------
def load_vision() -> pd.DataFrame:
    """Load the provided vision-model predictions (PENet) and study metadata."""
    df = pd.read_csv(C.DATA_DIR / C.VISION_FILE)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    required = {C.IDX_COL, C.LABEL_COL, C.VISION_PRED_COL, C.SPLIT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Vision.csv missing expected columns: {missing}")
    return df.set_index(C.IDX_COL)


def _load_ehr_table(name: str) -> pd.DataFrame:
    df = pd.read_csv(C.DATA_DIR / name)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    if C.IDX_COL not in df.columns:
        raise ValueError(f"{name} has no '{C.IDX_COL}' column to join on")
    return df.set_index(C.IDX_COL)


def build_ehr_features(train_idx: pd.Index | None = None) -> pd.DataFrame:
    """Merge the EHR tables on `idx` and apply the dataset's feature protocol.

    Rare-feature removal is fit on the training rows only (`train_idx`) to avoid using test
    information; if `train_idx` is None, all rows are used (call again with the real split for a
    leak-free feature set).
    """
    tables = [_load_ehr_table(name) for name in C.EHR_FILES]
    ehr = reduce(lambda a, b: a.join(b, how="outer"), tables)

    # coerce to numeric; RadFusion EHR features are numeric indicators / counts
    ehr = ehr.apply(pd.to_numeric, errors="coerce")

    # de-duplicate identical columns and drop all-NaN columns
    ehr = ehr.loc[:, ~ehr.columns.duplicated()]
    ehr = ehr.dropna(axis=1, how="all")

    # missing indicator/count values -> 0 (absence), matching one-hot / count semantics
    ehr = ehr.fillna(0.0)

    # remove rare columns (present in <1% of training rows) and zero-variance columns
    fit_rows = ehr.loc[ehr.index.intersection(train_idx)] if train_idx is not None else ehr
    nonzero_frac = (fit_rows != 0).mean(axis=0)
    keep = nonzero_frac[nonzero_frac >= C.RARE_ICD_THRESHOLD].index
    ehr = ehr[keep]
    ehr = ehr.loc[:, ehr.std(axis=0) > 0]

    return ehr


def assemble_dataset() -> pd.DataFrame:
    """Return a single frame indexed by `idx` with EHR features + provided vision score + metadata."""
    vision = load_vision()
    train_idx = vision.index[vision[C.SPLIT_COL] == "train"]
    ehr = build_ehr_features(train_idx=train_idx)

    df = ehr.join(vision[[C.LABEL_COL, C.VISION_PRED_COL, C.SPLIT_COL, C.PE_TYPE_COL]], how="inner")
    df = df.dropna(subset=[C.LABEL_COL, C.VISION_PRED_COL, C.SPLIT_COL])
    df[C.LABEL_COL] = df[C.LABEL_COL].astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """EHR feature column names (everything except label / vision / split / metadata)."""
    reserved = {C.LABEL_COL, C.VISION_PRED_COL, C.SPLIT_COL, C.PE_TYPE_COL,
                C.EMR_PRED_COL, C.EMR_PRED_SVC_COL, C.FUSION_AVG_COL, C.FUSION_MAX_COL}
    return [c for c in df.columns if c not in reserved]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RadFusion data utilities")
    ap.add_argument("--download", action="store_true", help="download the CSVs into ./data")
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()
    if args.download:
        download(force=args.force)
    else:
        d = assemble_dataset()
        print(f"assembled {d.shape[0]} studies x {len(feature_columns(d))} EHR features")
        print(d[C.SPLIT_COL].value_counts())
