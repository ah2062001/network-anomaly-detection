
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

a = pd.read_csv(ROOT / "data/raw/ddos.csv", low_memory=False)
b = pd.read_csv(
    ROOT / "data/raw/cicddos2019/cicddos2019_dataset.csv",
    low_memory=False
)

a.columns = a.columns.str.strip()
b.columns = b.columns.str.strip()

a["Label"] = a["Label"].astype(str).str.strip()
b["Class"] = b["Class"].astype(str).str.strip()

# Compare benign with benign, attack with attack.
a["BinaryClass"] = a["Label"].map({
    "BENIGN": "Benign",
    "DDoS": "Attack"
})

b["BinaryClass"] = b["Class"]

shared = sorted(set(a.columns) & set(b.columns))

shared = [
    col for col in shared
    if col not in {"Label", "Class", "BinaryClass"}
    and pd.api.types.is_numeric_dtype(a[col])
    and pd.api.types.is_numeric_dtype(b[col])
]

rows = []

for col in shared:
    for cls in ["Benign", "Attack"]:
        x = pd.to_numeric(
            a.loc[a["BinaryClass"] == cls, col],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        y = pd.to_numeric(
            b.loc[b["BinaryClass"] == cls, col],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(x) == 0 or len(y) == 0:
            continue

        med_a = x.median()
        med_b = y.median()

        # Relative difference; descriptive only.
        scale = max(abs(med_a), abs(med_b), 1.0)
        difference = abs(med_a - med_b) / scale

        rows.append({
            "Feature": col,
            "Class": cls,
            "Median_2017": med_a,
            "Median_2019": med_b,
            "Relative_Median_Difference": difference,
            "Zero_Fraction_2017": (x == 0).mean(),
            "Zero_Fraction_2019": (y == 0).mean(),
        })

result = pd.DataFrame(rows)

result = result.sort_values(
    "Relative_Median_Difference",
    ascending=False
)

out = ROOT / "reports/feature_shift_audit.csv"
out.parent.mkdir(exist_ok=True)
result.to_csv(out, index=False)

print("\n=== LARGEST MEDIAN DIFFERENCES ===")
print(result.head(25).to_string(index=False))

print("\nSaved:", out)
