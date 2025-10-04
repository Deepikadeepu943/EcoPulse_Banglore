import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os

# -----------------------
# 1. Load State Waste Data
# -----------------------
waste_file = "AllStates_WasteData.csv"
if os.path.exists(waste_file):
    waste_data = pd.read_csv(waste_file)
    print(f"✅ Loaded: {waste_file}, shape = {waste_data.shape}")
else:
    raise FileNotFoundError(f"{waste_file} not found.")

# Clean column names
waste_data.columns = [c.strip().replace(" ", "_") for c in waste_data.columns]

# Check required columns
required_cols = ["SolidwasteGenerated_TPD", "Treated_TPD", "State"]
for col in required_cols:
    if col not in waste_data.columns:
        raise KeyError(f"Required column missing: {col}")

# Create RiskLevel column
waste_data["RiskLevel"] = waste_data.apply(
    lambda row: "High" if (row["SolidwasteGenerated_TPD"] - row["Treated_TPD"]) > 0.5 * row["SolidwasteGenerated_TPD"]
    else "Low", axis=1
)

# -----------------------
# 2. Waste Visualization
# -----------------------
# Bar chart: Generated vs Treated
plt.figure(figsize=(12, 6))
plt.bar(waste_data["State"], waste_data["SolidwasteGenerated_TPD"], label="Generated TPD", alpha=0.7, color="orange")
plt.bar(waste_data["State"], waste_data["Treated_TPD"], label="Treated TPD", alpha=0.7, color="green")
plt.xlabel("State")
plt.ylabel("Tonnes per Day (TPD)")
plt.title("Solid Waste Generated vs Treated per State")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# Pie chart: Risk Level
plt.figure(figsize=(6, 6))
waste_data["RiskLevel"].value_counts().plot.pie(
    autopct='%1.1f%%', colors=["red", "green"], explode=[0.05, 0.05]
)
plt.title("Risk Level Distribution Across States")
plt.show()

# Optional: Individual state bar charts
for state in waste_data["State"].unique():
    state_data = waste_data[waste_data["State"] == state]
    plt.figure(figsize=(5, 4))
    plt.bar(["Generated", "Treated"], [
        state_data["SolidwasteGenerated_TPD"].values[0],
        state_data["Treated_TPD"].values[0]
    ], color=["orange", "green"])
    plt.title(f"{state} - Waste Generated vs Treated")
    plt.ylabel("TPD")
    plt.show()

# -----------------------
# 3. ML Risk Prediction
# -----------------------
X = waste_data[["SolidwasteGenerated_TPD", "Treated_TPD"]]
y = waste_data["RiskLevel"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("ML Model Accuracy:", model.score(X_test, y_test))

sample = [[12000, 6000]]
prediction = model.predict(sample)
print("Predicted Risk Level for sample:", prediction[0])

