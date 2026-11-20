import os
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


# ---------- Brief program ----------
def heat_balance(T):
    return T - 25 - 10 * np.exp(-T / 20)


solution = root_scalar(heat_balance, bracket=[20, 40], method="bisect", xtol=1e-10)


def g(T):
    return 25 + 10 * np.exp(-T / 20)


T = 30.0
history = [T]
for _ in range(100):
    T_new = g(T)
    history.append(T_new)
    if abs(T_new - T) < 1e-10:
        break
    T = T_new

print(f"Bisection equilibrium temperature: {solution.root:.6f} C")
print("Bisection iterations:", solution.iterations)
print(f"Fixed-point equilibrium temperature: {history[-1]:.6f} C")
print("Fixed-point iterations:", len(history) - 1)

os.makedirs("results", exist_ok=True)
temps = np.linspace(20, 40, 300)
plt.plot(temps, heat_balance(temps))
plt.axhline(0, linewidth=0.8)
plt.axvline(solution.root, linestyle="--")
plt.xlabel("Temperature (C)")
plt.ylabel("Heat-balance function")
plt.title("Equilibrium temperature")
plt.grid(True)
plt.savefig("results/task_10.png", dpi=150)
plt.show()

# ---------- Changed parameters: tolerance and bracket ----------
for xtol in (1e-3, 1e-12):
    r = root_scalar(heat_balance, bracket=[20, 40], method="bisect", xtol=xtol)
    print(f"\n[Changed] xtol={xtol}: {r.iterations} iterations")
r = root_scalar(heat_balance, bracket=[0, 100], method="bisect", xtol=1e-10)
print("[Changed] bracket [0, 100]:", r.iterations, "iterations")

# ---------- Modified case: business break-even points ----------
# Assumptions: revenue 45 q; cost 12000 + 20 q + 0.002 q^2
profit = lambda q: 45 * q - (12000 + 20 * q + 0.002 * q ** 2)
low = root_scalar(profit, bracket=[100, 1000], method="bisect").root
high = root_scalar(profit, bracket=[5000, 12000], method="bisect").root
print(f"\n[Modified] break-even quantities: {low:.1f} and {high:.1f} units")
