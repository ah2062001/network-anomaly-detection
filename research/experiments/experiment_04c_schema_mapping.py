from pathlib import Path
import numpy as np
import pandas as pd

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

mapping = {
    "Average Packet Size": "Avg Packet Size",
    "Max Packet Length": "Packet Length Max",
    "Min Packet Length": "Packet Length Min",
    "Total Length of Fwd Packets": "Fwd Packets Length Total",
    "Total Length of Bwd Packets": "Bwd Packets Length Total",
    "Init_Win_bytes_forward": "Init Fwd Win Bytes",
    "Init_Win_bytes_backward": "Init Bwd Win Bytes",
    "act_data_pkt_fwd": "Fwd Act Data Packets",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

def numeric(df, name):
    return pd.to_numeric(
        df[name], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

records = []

for source_col, target_col in mapping.items():
    if source_col not in source or target_col not in target:
        print("MISSING:", source_col, "or", target_col)
        continue

    for dataset, df, col, label_col, attack_value in [
        ("2017", source, source_col, "Label", "DDoS"),
        ("2019", target, target_col, "Class", "Attack")
    ]:
        values = numeric(df, col)
        is_attack = df[label_col].astype(str).str.strip().eq(
            attack_value
        )

        for class_name, mask in [
            ("Attack", is_attack),
            ("Benign", ~is_attack)
        ]:
            v = values[mask]
            records.append({
                "canonical_feature": source_col,
                "dataset": dataset,
                "class": class_name,
                "count": len(v),
                "median": v.median(),
                "zero_fraction": v.eq(0).mean(),
                "missing_fraction": v.isna().mean(),
                "p90": v.quantile(0.90),
            })

result = pd.DataFrame(records)
result.to_csv(OUT / "candidate_alias_profiles.csv", index=False)

print("\n=== CANDIDATE ALIAS PROFILES: ATTACK TRAFFIC ===")
print(result[result["class"] == "Attack"].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"
))

# Compare similarly defined features inside each original dataset.
pairs = [
    ("Total Length of Fwd Packets", "Subflow Fwd Bytes"),
    ("Total Length of Bwd Packets", "Subflow Bwd Bytes"),
    ("Fwd Packets Length Total", "Subflow Fwd Bytes"),
    ("Bwd Packets Length Total", "Subflow Bwd Bytes"),
    ("Fwd Packet Length Mean", "Avg Fwd Segment Size"),
    ("Bwd Packet Length Mean", "Avg Bwd Segment Size"),
]

checks = []
for dataset_name, df in [("2017", source), ("2019", target)]:
    for a_name, b_name in pairs:
        if a_name not in df or b_name not in df:
            continue

        a = numeric(df, a_name)
        b = numeric(df, b_name)
        valid = a.notna() & b.notna()

        if not valid.any():
            continue

        matches = np.isclose(
            a[valid].to_numpy(),
            b[valid].to_numpy(),
            rtol=1e-6,
            atol=1e-6
        )

        checks.append({
            "dataset": dataset_name,
            "feature_a": a_name,
            "feature_b": b_name,
            "valid_rows": int(valid.sum()),
            "fraction_equal": float(matches.mean())
        })

checks_df = pd.DataFrame(checks)
checks_df.to_csv(OUT / "alias_relationship_checks.csv", index=False)

print("\n=== INTERNAL RELATIONSHIP CHECKS ===")
print(checks_df.to_string(index=False))

print("\nReports saved to:", OUT)
print("Matching values do not by themselves prove")
print("equivalent feature-extraction semantics.")
