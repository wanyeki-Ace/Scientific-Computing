import numpy as np
from scipy.integrate import simpson

# ---------- Brief program (needs NumPy 2.0+ for np.trapezoid) ----------
time_hours = np.array([0, 2, 4, 6, 8, 10, 12], dtype=float)
power_kw = np.array([0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9])

energy_trapezoid = np.trapezoid(power_kw, time_hours)
energy_simpson = simpson(power_kw, x=time_hours)

print(f"Energy using trapezoidal rule: {energy_trapezoid:.3f} kWh")
print(f"Energy using Simpson's rule: {energy_simpson:.3f} kWh")
print(f"Difference between estimates: {abs(energy_simpson-energy_trapezoid):.3f} kWh")

# ---------- Changed parameters ----------
scaled = power_kw * 1.2
print("\n[Changed] power x1.2:", np.trapezoid(scaled, time_hours), simpson(scaled, x=time_hours))
xs = np.linspace(0, np.pi, 5)
print("[Changed] sin(x) 0..pi, exact 2:", np.trapezoid(np.sin(xs), xs), simpson(np.sin(xs), x=xs))

# ---------- Modified case: distance from velocity, and energy cost ----------
# Assumptions: velocity in m/s; electricity tariff KES 24 per kWh (illustrative)
t = np.arange(0, 11, 2.0)
v = np.array([0, 5, 9, 12, 14, 15.0])
print("\n[Modified] distance (m):", np.trapezoid(v, t), simpson(v, x=t))
print("[Modified] cost of 12-hour energy (KES):", round(energy_simpson * 24, 1))
