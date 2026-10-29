
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(ROOT / "models/random_forest.joblib")

# Get trained Random Forest from pipeline
rf = model.named_steps["classifier"]

# Recover feature names from the fitted imputer
features = model.named_steps["imputer"].get_feature_names_out()

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 20 Important Features:")
print(importance.head(20).to_string(index=False))

# Save results
output_dir = ROOT / "reports"
output_dir.mkdir(exist_ok=True)

importance.to_csv(
    output_dir / "feature_importance.csv",
    index=False
)

# Plot top 15
top = importance.head(15).sort_values("Importance")

plt.figure(figsize=(10, 7))
plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Feature Importance")
plt.title("Random Forest - Top 15 Network Traffic Features")
plt.tight_layout()

plt.savefig(
    output_dir / "feature_importance.png",
    dpi=200
)

plt.show()
