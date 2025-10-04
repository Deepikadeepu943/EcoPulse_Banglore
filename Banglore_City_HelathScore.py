import pandas as pd
import matplotlib.pyplot as plt

# Example: manually insert your computed Risk values (replace with real values)
# Waste Risk (1 - Treated/Generated) from your Waste script
waste_risk = 0.63   # 63% untreated

# Water Risk (1 - WQI) from your Water script
water_risk = 0.45   # example, replace with your WQI result

# Climate Risk (AirRiskIndex average from climate yearly data)
climate_risk = 0.55 # example, replace with average AirRiskIndex

# Put into DataFrame
df = pd.DataFrame({
    "Domain":["Waste","Water","Climate"],
    "RiskIndex":[waste_risk,water_risk,climate_risk]
})

print(df)

# Plot
plt.figure(figsize=(7,5))
plt.bar(df["Domain"],df["RiskIndex"],color=["red","blue","green"])
plt.ylim(0,1)
plt.ylabel("Risk Index (0=Low, 1=High)")
plt.title("Bangalore City Health Risk by Domain")
plt.tight_layout()
plt.savefig("bangalore_city_health_score.png",dpi=150)
plt.show()
