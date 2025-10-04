import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---- Load Data ----
data = {
    "State": ["Karnataka", "Maharashtra", "Delhi"],
    "SolidWasteGenerated_TPD": [12258, 22570, 10471],
    "Collected_TPD": [10011, 22779, 10467],
    "Treated_TPD": [4489, 16037, 5194],
    "LandfillSites_Identified": [199, 18, 1],
    "LandfillSites_Operational": [191, None, None],
    "Dumpsites": [None, 327, 5],
    "Waste_to_Energy_Plants": [0, 1, 3],
    "Authorizations_Granted": [0, 56, 0],
    "Performance_Rank": [27, 2, 11],
    "Performance_Score": [41, 70.5, 54.5]
}
df = pd.DataFrame(data)

# ---- Risk Index ----
df["Risk_Index"] = 1 - (df["Treated_TPD"] / df["SolidWasteGenerated_TPD"])

print("\nDataset with Risk Index:\n", df[["State","SolidWasteGenerated_TPD","Treated_TPD","Risk_Index"]])

# ---- Combined Figure ----
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Waste Generated vs Treated
axs[0].bar(df["State"], df["SolidWasteGenerated_TPD"], label="Generated", alpha=0.7)
axs[0].bar(df["State"], df["Treated_TPD"], label="Treated", alpha=0.7)
axs[0].set_title("Waste Generated vs Treated")
axs[0].set_ylabel("TPD")
axs[0].legend()

# Plot 2: Performance Score
axs[1].bar(df["State"], df["Performance_Score"], color="green")
axs[1].set_title("Performance Score")
axs[1].set_ylabel("Score")

# Plot 3: Risk Index
axs[2].bar(df["State"], df["Risk_Index"], color="red")
axs[2].set_title("Risk Index (Higher = More Risk)")
axs[2].set_ylabel("Risk Index")

plt.tight_layout()
plt.savefig("Waste_Combined_Plots.png")
plt.show()

# ---- ML Model ----
X = df[["SolidWasteGenerated_TPD","Collected_TPD","Treated_TPD",
        "LandfillSites_Identified","Waste_to_Energy_Plants","Authorizations_Granted"]].fillna(0)
y = df["Performance_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nFeature Importance (what drives performance score):")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.3f}")

print("\nModel Accuracy (R²):", model.score(X_train, y_train))
