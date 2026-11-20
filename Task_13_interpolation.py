import os
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

# ---------- Brief program ----------
hours = np.array([8, 9, 10, 11, 12, 13, 14], dtype=float)
temperature = np.array([18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0])

linear = interp1d(hours, temperature, kind="linear")
cubic = CubicSpline(hours, temperature)

ten_minute_times = np.arange(8, 14 + 1 / 6, 1 / 6)
linear_estimates = linear(ten_minute_times)
cubic_estimates = cubic(ten_minute_times)

target_time = 10.5
print(f"Linear estimate at 10:30: {float(linear(target_time)):.2f} C")
print(f"Cubic estimate at 10:30: {float(cubic(target_time)):.2f} C")

os.makedirs("results", exist_ok=True)
plt.scatter(hours, temperature, label="Hourly measurements")
plt.plot(ten_minute_times, linear_estimates, label="Linear")
plt.plot(ten_minute_times, cubic_estimates, label="Cubic spline")
plt.xlabel("Hour of day")
plt.ylabel("Temperature (C)")
plt.title("Temperature interpolation")
plt.legend()
plt.grid(True)
plt.savefig("results/task_13.png", dpi=150)
plt.show()

# ---------- Changed parameter: 11:00 reading becomes 30 C ----------
spike = temperature.copy()
spike[3] = 30.0
lin2, cub2 = interp1d(hours, spike), CubicSpline(hours, spike)
print("\n[Changed] 10:30 linear/cubic:", float(lin2(10.5)), float(cub2(10.5)))
print("[Changed] 11:30 linear/cubic:", float(lin2(11.5)), float(cub2(11.5)))
print("[Changed] cubic maximum:", cub2(np.linspace(8, 14, 601)).max())
print("[Changed] cubic extrapolated to 15:00:", float(cubic(15.0)))

# ---------- Modified case: river gauge missed day 3 ----------
d = np.array([0, 1, 2, 4, 5, 6.0])
level = np.array([10, 12, 15, 22, 25, 24.0])
print("\n[Modified] day 3 linear:", float(interp1d(d, level)(3)),
      "cubic:", float(CubicSpline(d, level)(3)))
