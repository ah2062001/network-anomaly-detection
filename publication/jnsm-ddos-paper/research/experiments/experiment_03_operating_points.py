
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/score_diagnostics"

source = pd.read_csv(BASE / "source_holdout_scores.csv.gz")
target = pd.read_csv(BASE / "external_per_flow_scores.csv.gz")

THRESHOLD = 0.01

def evaluate(name, data):
    y = data["true_attack"].to_numpy(int)
    s = data["attack_probability"].to_numpy(float)

    print(f"\n=== {name} ===")

    benign = s[y == 0]
    attack = s[y == 1]

    for label, values in [
        ("Benign", benign),
        ("Attack", attack),
    ]:
        print(
            f"{label}: {len(values):,} flows | "
            f"score=0.01: "
            f"{np.mean(np.isclose(values, THRESHOLD)):.2%} | "
            f"score=0: {np.mean(values == 0):.2%}"
        )

    rows = []

    for threshold, inclusive in [
        (0.00, False),
        (0.01, False),
        (0.01, True),
        (0.02, False),
        (0.03, False),
        (0.05, False),
        (0.10, False),
        (0.50, True),
    ]:
        flagged = s >= threshold if inclusive else s > threshold

        recall = flagged[y == 1].mean()
        fpr = flagged[y == 0].mean()
        precision = (
            y[flagged].mean() if flagged.sum() else 0.0
        )

        rows.append({
            "dataset": name,
            "rule": (">=" if inclusive else ">") + str(threshold),
            "attack_recall": recall,
            "benign_fpr": fpr,
            "precision": precision,
            "false_positives": int(flagged[y == 0].sum()),
            "detected_attacks": int(flagged[y == 1].sum()),
        })

    result = pd.DataFrame(rows)

    print("\nObserved threshold behavior:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    return result

source_result = evaluate("SOURCE HOLDOUT", source)
target_result = evaluate("EXTERNAL TARGET", target)

# Target-label-informed ROC operating points.
# Target-label-informed thresholds are diagnostic oracle estimates, not
# source-calibrated operating points.
y = target["true_attack"].to_numpy(int)
s = target["attack_probability"].to_numpy(float)

fpr, tpr, thresholds = roc_curve(
    y, s, drop_intermediate=False
)

print("\n=== TARGET ORACLE OPERATING POINTS ===")
print("DIAGNOSTIC ONLY — uses target labels")

oracle_rows = []

for budget in [0.01, 0.05, 0.10]:
    eligible = np.where(fpr <= budget + 1e-12)[0]
    idx = eligible[np.argmax(tpr[eligible])]

    oracle_rows.append({
        "maximum_fpr_budget": budget,
        "achieved_fpr": fpr[idx],
        "attack_recall": tpr[idx],
        "threshold_ge": thresholds[idx],
    })

oracle = pd.DataFrame(oracle_rows)
print(oracle.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

out = BASE / "operating_point_audit.csv"
pd.concat([source_result, target_result]).to_csv(out, index=False)
oracle.to_csv(BASE / "target_oracle_operating_points.csv", index=False)

print("\nSaved operating-point reports.")
