# water_quality_bangalore.py
# Run in PyCharm, Jupyter, or Colab.
# No external files needed — the sample data is embedded below.
# Outputs: water_wqi_by_site.png, water_feature_importance.png, water_scatter_do_ph.png
# Requirements:
# pip install pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 1. Input: sample rows parsed from your pasted 2021 table (small sample)
# You can add more rows in the same format if you have them.
# Station code, location, state, temp_min, temp_max, do_min, do_max, ph_min, ph_max,
# conductivity_min, conductivity_max, bod_min, bod_max, nitrate_min, nitrate_max,
# fecal_min, fecal_max, totalcol_min, totalcol_max, fecalstrep_min, fecalstrep_max
rows = [
    [30051, "RIVER GODAVARI AT BASER (BANGALORE)", "Karnataka", np.nan, np.nan, 7.0, 7.0, 7.0, 7.0, 680, 680, 2.0, 2.0, 7.0, 7.0, 6, 6, 26, 26, np.nan, np.nan],
    [None, "RIVER MANJEERA AT JAN WADA (BANGALORE)", "Karnataka", np.nan, np.nan, np.nan, np.nan, 8.0, 8.0, 498, 498, 3.0, 3.0, 0.32, 0.32, 170, 170, 920, 920, np.nan],
    [None, "RIVER KRISHNA AT DEODURGA (BANGALORE)", "Karnataka", 26.0, 26.0, 7.0, 7.0, 7.0, 7.0, 1290, 1290, 3.0, 3.0, 0.30, 0.30, 170, 170, 920, 920, np.nan],
    [None, "RIVER BHIMA AT GANGAPUR (BANGALORE)", "Karnataka", 29.0, 29.0, 7.0, 7.0, 8.0, 8.0, 1420, 1420, 3.0, 3.0, 0.32, 0.32, 700, 700, 3500, 3500, np.nan],
]

cols = ["station_code","location","state",
        "temp_min","temp_max",
        "do_min","do_max",
        "ph_min","ph_max",
        "cond_min","cond_max",
        "bod_min","bod_max",
        "nitrate_min","nitrate_max",
        "fecal_min","fecal_max",
        "totalcol_min","totalcol_max",
        "fecalstrep_min","fecalstrep_max"]

df = pd.DataFrame(rows, columns=cols)

# -------------------------
# 2. Preprocess: compute site-level averages (use min/max where available)
def avg_or_nan(a,b):
    if pd.isna(a) and pd.isna(b):
        return np.nan
    if pd.isna(a):
        return b
    if pd.isna(b):
        return a
    return (a+b)/2.0

df2 = pd.DataFrame()
df2["site"] = df["location"].str.replace("\n"," ").str.strip()
df2["state"] = df["state"]
df2["temp_c"] = df.apply(lambda r: avg_or_nan(r["temp_min"], r["temp_max"]), axis=1)
df2["do_mg_L"] = df.apply(lambda r: avg_or_nan(r["do_min"], r["do_max"]), axis=1)
df2["ph"] = df.apply(lambda r: avg_or_nan(r["ph_min"], r["ph_max"]), axis=1)
df2["conductivity_uS_cm"] = df.apply(lambda r: avg_or_nan(r["cond_min"], r["cond_max"]), axis=1)
df2["bod_mg_L"] = df.apply(lambda r: avg_or_nan(r["bod_min"], r["bod_max"]), axis=1)
df2["nitrate_mg_L"] = df.apply(lambda r: avg_or_nan(r["nitrate_min"], r["nitrate_max"]), axis=1)
df2["fecal_coliform"] = df.apply(lambda r: avg_or_nan(r["fecal_min"], r["fecal_max"]), axis=1)
df2["total_coliform"] = df.apply(lambda r: avg_or_nan(r["totalcol_min"], r["totalcol_max"]), axis=1)
df2["fecal_strep"] = df.apply(lambda r: avg_or_nan(r["fecalstrep_min"], r["fecalstrep_max"]), axis=1)

# show parsed table
print("Parsed water-quality sites (sample):")
print(df2[['site','temp_c','do_mg_L','ph','conductivity_uS_cm','bod_mg_L','nitrate_mg_L','fecal_coliform','total_coliform']])

# -------------------------
# 3. Compute Water Quality Index (simple, transparent approach)
# We'll normalize each indicator across sites: higher bad for conductivity, BOD, nitrate, fecal/total coliform.
# DO is better when higher; pH is best near 7 (distance from 7 is bad).
# We'll produce per-parameter normalized "subindex" -> average -> WQI (1 best, 0 worst).

def normalize_series(s, higher_is_worse=True):
    s_num = pd.to_numeric(s, errors='coerce')
    if s_num.isnull().all():
        return s_num * 0.0
    mn, mx = s_num.min(skipna=True), s_num.max(skipna=True)
    if pd.isnull(mn) or pd.isnull(mx) or mn==mx:
        return (s_num - mn) * 0.0
    norm = (s_num - mn) / (mx - mn)
    return norm if higher_is_worse else (1 - norm)

# subindices
dfw = df2.copy()
dfw["do_sub"] = normalize_series(dfw["do_mg_L"], higher_is_worse=False)   # higher DO better => lower risk
# pH: best at 7 -> distance from 7 is worse
ph_badness = (dfw["ph"] - 7.0).abs()
dfw["ph_sub"] = normalize_series(ph_badness, higher_is_worse=True)
dfw["cond_sub"] = normalize_series(dfw["conductivity_uS_cm"], higher_is_worse=True)
dfw["bod_sub"] = normalize_series(dfw["bod_mg_L"], higher_is_worse=True)
dfw["nitrate_sub"] = normalize_series(dfw["nitrate_mg_L"], higher_is_worse=True)
dfw["fecal_sub"] = normalize_series(dfw["fecal_coliform"], higher_is_worse=True)
dfw["totalcol_sub"] = normalize_series(dfw["total_coliform"], higher_is_worse=True)

# choose which subindices to include (ignore if all NaN)
subs = ["do_sub","ph_sub","cond_sub","bod_sub","nitrate_sub","fecal_sub","totalcol_sub"]
available_subs = [s for s in subs if dfw[s].notna().any()]
print("\nAvailable sub-indices:", available_subs)

# WQI = 1 - average(subindices)  (so higher WQI -> better water; Risk = 1-WQI)
dfw["sub_mean"] = dfw[available_subs].mean(axis=1, skipna=True)
dfw["WQI"] = 1.0 - dfw["sub_mean"]   # 1 best, 0 worst
dfw["Water_Risk"] = 1.0 - dfw["WQI"] # equal to sub_mean

# show WQI
print("\nComputed WQI and Risk (0-1 scale) per site:")
print(dfw[["site","WQI","Water_Risk"]])

# -------------------------
# 4. Save WQI plot and Risk plot
plt.figure(figsize=(9,4))
plt.bar(dfw["site"], dfw["WQI"], color='tab:blue', alpha=0.8)
plt.xticks(rotation=35, ha='right')
plt.ylabel("WQI (1 = best)")
plt.title("Water Quality Index (WQI) - Bangalore (sample sites)")
plt.tight_layout()
plt.savefig("water_wqi_by_site.png", dpi=150)
plt.show()

# Risk plot (1-WQI)
plt.figure(figsize=(9,4))
plt.bar(dfw["site"], dfw["Water_Risk"], color='tab:red', alpha=0.8)
plt.xticks(rotation=35, ha='right')
plt.ylabel("Water Risk (higher = worse)")
plt.title("Water Risk Index - Bangalore (sample sites)")
plt.tight_layout()
plt.savefig("water_risk_by_site.png", dpi=150)
plt.show()

# -------------------------
# 5. ML: predict WQI from raw features (Random Forest)
# We use numeric features only; fill NaN with median (simple approach for small dataset)
features = ["temp_c","do_mg_L","ph","conductivity_uS_cm","bod_mg_L","nitrate_mg_L","fecal_coliform","total_coliform"]
X = dfw[features].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())    # small dataset; median fill
y = dfw["WQI"].fillna(dfw["WQI"].mean())

# If not enough rows for test split, we still train and show feature importance
if len(dfw) >= 2:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print("\nFeature importance for predicting WQI (higher -> more important):")
    print(feat_imp)
    # save feature importance plot
    plt.figure(figsize=(7,4))
    feat_imp.plot(kind='bar', color='tab:green')
    plt.title("Feature Importance (predicting WQI)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("water_feature_importance.png", dpi=150)
    plt.show()
else:
    print("\nNot enough rows to train ML model - add more site rows to enable ML.")

# -------------------------
# 6. Helpful scatter plot (DO vs pH) to visualize dangerous points
plt.figure(figsize=(6,4))
plt.scatter(dfw["ph"], dfw["do_mg_L"], s=80, c=dfw["Water_Risk"], cmap='Reds', edgecolor='k')
for i,r in dfw.iterrows():
    plt.text(r["ph"]+0.02, r["do_mg_L"]+0.02, str(i+1), fontsize=9)
plt.xlabel("pH")
plt.ylabel("DO (mg/L)")
plt.title("DO vs pH (label points) — color = Water Risk")
plt.grid(True)
plt.tight_layout()
plt.savefig("water_scatter_do_ph.png", dpi=150)
plt.show()

# -------------------------
print("\nAll images saved: water_wqi_by_site.png, water_risk_by_site.png, water_feature_importance.png, water_scatter_do_ph.png")
print("\nIf you have additional site rows (more monitoring locations), add them to `rows` list at top and re-run for stronger ML.")
