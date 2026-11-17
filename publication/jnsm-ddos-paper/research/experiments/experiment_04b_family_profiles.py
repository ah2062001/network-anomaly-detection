from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/feature_semantics"
OUT.mkdir(parents=True, exist_ok=True)

source = pd.read_csv(ROOT / "data/raw/ddos.csv", low_memory=False)
target = pd.read_csv(
    ROOT / "data/raw/cicddos2019/cicddos2019_dataset.csv",
    low_memory=False
)

source.columns = source.columns.str.strip()
target.columns = target.columns.str.strip()

scols = set(source.columns)
tcols = set(target.columns)

inventory = pd.DataFrame({
    "feature": sorted(scols | tcols)
})
inventory["in_2017"] = inventory.feature.isin(scols)
inventory["in_2019"] = inventory.feature.isin(tcols)
inventory.to_csv(OUT / "schema_coverage.csv", index=False)

print("\n=== DIFFERENT COLUMN NAMES ===")
print("\nOnly 2017:")
print(sorted(scols - tcols))
print("\nOnly 2019:")
print(sorted(tcols - scols))

features = [
    "Fwd Packet Length Mean",
    "Fwd Packet Length Min",
    "Fwd Packet Length Max",
    "Subflow Fwd Bytes",
    "Bwd Packet Length Mean",
    "Total Backward Packets",
    "Flow Duration",
    "Flow IAT Min",
    "Fwd IAT Min",
]

available = [f for f in features if f in target.columns]

for feature in available:
    target[feature] = pd.to_numeric(
        target[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

records = []

for family, group in target.groupby("Label", dropna=False):
    row = {
        "attack_family": str(family),
        "count": len(group),
        "attack_fraction": (
            group["Class"].astype(str).str.strip().str.lower()
            .eq("attack").mean()
        ),
    }

    for feature in available:
        values = group[feature]
        row[feature + "_median"] = values.median()
        row[feature + "_zero_fraction"] = (
            values.eq(0).mean()
        )

    records.append(row)

profiles = (
    pd.DataFrame(records)
    .sort_values("count", ascending=False)
)

profiles.to_csv(
    OUT / "attack_family_feature_profiles.csv",
    index=False
)

selected = [
    "attack_family",
    "count",
    "Fwd Packet Length Mean_median",
    "Bwd Packet Length Mean_zero_fraction",
    "Subflow Fwd Bytes_median",
]

selected = [c for c in selected if c in profiles.columns]

print("\n=== FAMILY FEATURE PROFILES ===")
print(
    profiles[selected].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nReports saved:", OUT)
