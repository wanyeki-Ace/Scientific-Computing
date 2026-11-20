import numpy as np

# ---------- Brief program: daily temperature statistics ----------
temperatures_c = np.array([
    18.5, 18.0, 17.8, 17.5, 17.9, 19.0,
    21.5, 23.8, 25.6, 27.1, 28.3, 29.0,
    29.4, 29.1, 28.5, 27.6, 26.2, 24.8,
    23.4, 22.1, 21.0, 20.2, 19.6, 19.0
])

mean_temp = np.mean(temperatures_c)
min_temp = np.min(temperatures_c)
max_temp = np.max(temperatures_c)
temperatures_f = 1.8 * temperatures_c + 32

print(f"Daily mean temperature: {mean_temp:.2f} C")
print(f"Minimum temperature: {min_temp:.2f} C")
print(f"Maximum temperature: {max_temp:.2f} C")
print("First six Fahrenheit readings:", temperatures_f[:6])

# ---------- Changed parameter: every reading +5 C ----------
hot = temperatures_c + 5
print("\n[Changed] +5 C mean/min/max:", hot.mean(), hot.min(), hot.max())

# ---------- Modified case: hourly rainfall at a weather station ----------
# Assumptions: 24 readings (00:00-23:00) in mm; inches = mm / 25.4
rain_mm = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.5, 0, 0, 0, 1.2,
                    3.4, 5.0, 2.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.0])
print("\n[Modified] Total rain:", rain_mm.sum(), "mm")
print("[Modified] Peak:", rain_mm.max(), "mm at hour", rain_mm.argmax())
print("[Modified] Rainy hours:", (rain_mm > 0).sum())
print("[Modified] Peak in inches:", round(rain_mm.max() / 25.4, 3))
