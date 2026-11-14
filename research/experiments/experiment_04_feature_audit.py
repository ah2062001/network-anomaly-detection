from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/feature_semantics"
OUT.mkdir(parents=True, exist_ok=True)

source_path = ROOT / "data/raw/ddos.csv"
target_path = ROOT / "data/raw/cicddos2019/cicddos2019_dataset.csv"

print("Loading datasets...")
source = pd.read_csv(source_path, low_memory=False)
target = pd.read_csv(target_path, low_memory=False)

source.columns = source.columns.str.strip()
target.columns = target.columns.str.strip()

print("\nSource shape:", source.shape)
print("Target shape:", target.shape)
print("\nSource labels:")
print(source["Label"].value_counts().to_string())
print("\nTarget binary labels:")
print(target["Class"].value_counts().to_string())

# Keep the label mappings explicit.
source["audit_attack"] = (
    source["Label"].astype(str).str.strip().str.lower() == "ddos"
)
target["audit_attack"] = (
    target["Class"].astype(str).str.strip().str.lower() == "attack"
)

# Balanced diagnostic samples; never used to fit a detector.
def sample_by_class(df, n=20000):
    parts = []
    for _, group in df.groupby("audit_attack"):
        parts.append(
            group.sample(n=min(n, len(group)), random_state=42)
        )
    return pd.concat(parts, ignore_index=True)

s = sample_by_class(source)
t = sample_by_class(target)

exclude = {"Label", "Class", "audit_attack",
           "Flow ID", "Source IP", "Destination IP",
           "Timestamp"}

shared = sorted((set(s.columns) & set(t.columns)) - exclude)

print("\nShared column names:", len(shared))

records = []
numeric_features = []

for feature in shared:
    sx = pd.to_numeric(s[feature], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    tx = pd.to_numeric(t[feature], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    # Do not mislabel non-numeric identifiers as flow measurements.
    if sx.notna().mean() < 0.90 or tx.notna().mean() < 0.90:
        continue

    numeric_features.append(feature)

    for class_value, class_name in [
        (False, "Benign"), (True, "Attack")
    ]:
        a = sx[s["audit_attack"] == class_value].dropna()
        b = tx[t["audit_attack"] == class_value].dropna()

        if len(a) == 0 or len(b) == 0:
            continue

        ks = ks_2samp(a.to_numpy(), b.to_numpy()).statistic

        records.append({
            "feature": feature,
            "class": class_name,
            "source_median": float(a.median()),
            "target_median": float(b.median()),
            "source_zero_fraction": float((a == 0).mean()),
            "target_zero_fraction": float((b == 0).mean()),
            "source_missing_fraction": float(
                sx[s["audit_attack"] == class_value].isna().mean()
            ),
            "target_missing_fraction": float(
                tx[t["audit_attack"] == class_value].isna().mean()
            ),
            "ks_statistic": float(ks),
        })

audit = pd.DataFrame(records)
audit.to_csv(OUT / "feature_distribution_audit.csv", index=False)

print("\nShared numeric columns:", len(numeric_features))

for name in ["Attack", "Benign"]:
    print(f"\n=== {name.upper()} — LARGEST DISTRIBUTION SHIFTS ===")
    top = (
        audit[audit["class"] == name]
        .sort_values("ks_statistic", ascending=False)
        .head(12)
    )

    display_cols = [
        "feature", "source_median", "target_median",
        "source_zero_fraction", "target_zero_fraction",
        "ks_statistic"
    ]
    print(top[display_cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

# Sanity-check feature relationships within each dataset.
# These pairs commonly describe related quantities in CICFlowMeter.
# Disagreement is a flag for manual investigation, not proof of a bug.
pairs = [
    ("Fwd Packet Length Mean", "Avg Fwd Segment Size"),
    ("Bwd Packet Length Mean", "Avg Bwd Segment Size"),
    ("Total Length of Fwd Packets", "Subflow Fwd Bytes"),
    ("Total Length of Bwd Packets", "Subflow Bwd Bytes"),
]

identity_rows = []

for dataset_name, df in [("CIC-IDS2017", s), ("CIC-DDoS2019", t)]:
    for first, second in pairs:
        if first not in df.columns or second not in df.columns:
            continue

        a = pd.to_numeric(df[first], errors="coerce")
        b = pd.to_numeric(df[second], errors="coerce")

        valid = (
            np.isfinite(a.to_numpy(dtype=float)) &
            np.isfinite(b.to_numpy(dtype=float))
        )

        if not valid.any():
            continue

        aa = a.to_numpy(dtype=float)[valid]
        bb = b.to_numpy(dtype=float)[valid]

        identity_rows.append({
            "dataset": dataset_name,
            "feature_a": first,
            "feature_b": second,
            "valid_rows": int(valid.sum()),
            "fraction_nearly_equal": float(
                np.isclose(aa, bb, rtol=1e-6, atol=1e-6).mean()
            ),
            "median_absolute_difference": float(
                np.median(np.abs(aa - bb))
            ),
        })

identities = pd.DataFrame(identity_rows)
identities.to_csv(OUT / "feature_relationships.csv", index=False)

print("\n=== FEATURE RELATIONSHIP CHECKS ===")
if len(identities):
    print(identities.to_string(index=False))
else:
    print("No configured feature pairs were available in both datasets.")

print("\nIMPORTANT:")
print("Distribution shift is not proof of incompatible feature definitions.")
print("Feature definitions, units, and exporter versions still require")
print("manual verification against dataset provenance and documentation.")

print("\nReports saved in:", OUT)
