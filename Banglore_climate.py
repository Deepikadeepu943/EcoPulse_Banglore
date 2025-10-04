import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import re

# --------------------
# File dictionary (your files)
files = {
    2020: "Raw_data_1Day_2020_site_1558_Silk_Board_Bengaluru_KSPCB_1Day.csv",
    2021: "Raw_data_1Day_2021_site_1558_Silk_Board_Bengaluru_KSPCB_1Day.csv",
    2022: "Raw_data_1Day_2022_site_1558_Silk_Board_Bengaluru_KSPCB_1Day.csv",
    2023: "Raw_data_1Day_2023_site_1558_Silk_Board_Bengaluru_KSPCB_1Day.csv",
    2024: "Raw_data_1Day_2024_site_1558_Silk_Board_Bengaluru_KSPCB_1Day.csv"
}

# Helpers
def normalize_header(s):
    s = str(s)
    s = s.replace("Â", "").replace("µ","u").replace("°","")
    s = re.sub(r"[^a-zA-Z0-9]", "", s.lower())
    return s

def find_col(columns, keywords):
    norm = {c: normalize_header(c) for c in columns}
    for c,n in norm.items():
        for kw in keywords:
            if kw in n: return c
    return None

KEYS = {
    "PM2.5":["pm25"], "PM10":["pm10"], "NO2":["no2"], "Ozone":["ozone","o3"],
    "Temperature":["at","temp"], "Humidity":["rh","humidity"],
    "WindSpeed":["ws","wind"], "Rainfall":["rf","rain"]
}

# Collect yearly averages
dfs = []
for year,f in files.items():
    df = pd.read_csv(f, low_memory=False)
    row={"Year":year}
    for var,kws in KEYS.items():
        col = find_col(df.columns,kws)
        if col: row[var]=pd.to_numeric(df[col],errors="coerce").mean()
        else: row[var]=None
    dfs.append(row)
yearly=pd.DataFrame(dfs)

# Air Risk Index
pollutants=[c for c in ["PM2.5","PM10","NO2","Ozone"] if c in yearly]
for c in pollutants:
    yearly[c+"_norm"]=(yearly[c]-yearly[c].min())/(yearly[c].max()-yearly[c].min())
yearly["AirRiskIndex"]=yearly[[c+"_norm" for c in pollutants]].mean(axis=1)

# ML Feature Importance
feat_importance=None
if "PM2.5" in yearly and all(v in yearly for v in ["Temperature","Humidity","WindSpeed","Rainfall"]):
    df_ml=yearly.dropna(subset=["PM2.5","Temperature","Humidity","WindSpeed","Rainfall"])
    if len(df_ml)>=3:
        X=df_ml[["Temperature","Humidity","WindSpeed","Rainfall"]]
        y=df_ml["PM2.5"]
        model=RandomForestRegressor(n_estimators=100,random_state=42)
        model.fit(X,y)
        feat_importance=pd.Series(model.feature_importances_,index=X.columns)

# ---- Combined Plot ----
fig,axs=plt.subplots(1,3,figsize=(18,5))

# 1. PM trends
if "PM2.5" in yearly: axs[0].plot(yearly["Year"],yearly["PM2.5"],marker="o",label="PM2.5")
if "PM10" in yearly: axs[0].plot(yearly["Year"],yearly["PM10"],marker="s",label="PM10")
axs[0].set_title("PM2.5 & PM10 Trend"); axs[0].set_xlabel("Year"); axs[0].set_ylabel("µg/m³"); axs[0].legend(); axs[0].grid()

# 2. Air Risk Index
axs[1].bar(yearly["Year"],yearly["AirRiskIndex"],color="crimson",alpha=0.7)
axs[1].set_title("Air Risk Index"); axs[1].set_xlabel("Year"); axs[1].set_ylabel("0=Low, 1=High")

# 3. Feature Importance
if feat_importance is not None:
    feat_importance.plot(kind="bar",color="green",ax=axs[2])
    axs[2].set_title("Feature Importance (Weather → PM2.5)")
    axs[2].set_ylabel("Importance")
else:
    axs[2].text(0.5,0.5,"Not enough data for ML",ha="center",va="center")

plt.tight_layout()
plt.savefig("climate_analysis_combined.png",dpi=150)
plt.show()

print("Saved: climate_analysis_combined.png")
