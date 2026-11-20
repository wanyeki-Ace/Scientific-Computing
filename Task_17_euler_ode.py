import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- Brief program: Newton's law of cooling, dT/dt = -k (T - T_ambient) ----------
T_ambient = 25.0
T_initial = 90.0
k = 0.12
h = 1.0            # minutes
end_time = 30.0


def cooling_rate(T):
    return -k * (T - T_ambient)


time = np.arange(0, end_time + h, h)
temperature = np.zeros_like(time)
temperature[0] = T_initial
for i in range(len(time) - 1):
    temperature[i + 1] = temperature[i] + h * cooling_rate(temperature[i])

exact = T_ambient + (T_initial - T_ambient) * np.exp(-k * time)

print(f"Euler temperature after 30 min: {temperature[-1]:.2f} C")
print(f"Exact temperature after 30 min: {exact[-1]:.2f} C")
print(f"Final absolute error: {abs(temperature[-1]-exact[-1]):.4f} C")

os.makedirs("results", exist_ok=True)
plt.plot(time, temperature, marker="o", label="Euler")
plt.plot(time, exact, label="Exact solution")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (C)")
plt.title("Cooling of a hot object")
plt.legend()
plt.grid(True)
plt.savefig("results/task_17.png", dpi=150)
plt.show()


# ---------- Changed parameters: step size, k, ambient, initial temperature ----------
def euler(h, k=0.12, Ti=90.0, Ta=25.0, end=30.0):
    tm = np.arange(0, end + h / 2, h)
    T = np.zeros_like(tm)
    T[0] = Ti
    for i in range(len(tm) - 1):
        T[i + 1] = T[i] + h * (-k * (T[i] - Ta))
    return tm, T, Ta + (Ti - Ta) * np.exp(-k * tm)


print("\n[Changed] step size study")
for hh in (5, 2, 1, 0.5, 0.1, 17):
    tm, T, ex = euler(hh)
    print(f"h={hh}: Euler {T[-1]:.2f}, final error {abs(T[-1]-ex[-1]):.3f}, "
          f"largest error {np.abs(T - ex).max():.2f}")
for label, kw in (("k=0.25", dict(k=0.25)), ("Ta=5", dict(Ta=5.0)), ("Ti=60", dict(Ti=60.0))):
    tm, T, ex = euler(1, **kw)
    print(f"[Changed] {label}: Euler {T[-1]:.2f}, exact {ex[-1]:.2f}")

# ---------- Modified case: logistic population growth ----------
# Assumptions: r=0.5 per hour, K=10000, P0=1000
def euler_logistic(h, P0=1000, r=0.5, K=10000, end=20):
    P = P0
    for _ in range(int(end / h)):
        P += h * r * P * (1 - P / K)
    return P


exact_p = 10000 / (1 + 9 * np.exp(-0.5 * 20))
print(f"\n[Modified] h=1: {euler_logistic(1):.2f}, h=0.1: {euler_logistic(0.1):.2f}, "
      f"exact: {exact_p:.2f}")
