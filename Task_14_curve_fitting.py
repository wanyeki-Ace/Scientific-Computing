import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---------- Brief program ----------
users = np.array([50, 100, 150, 200, 250, 300, 350], dtype=float)
response_ms = np.array([82, 95, 115, 145, 185, 238, 305], dtype=float)


def model(x, a, b, c):
    return a * x ** 2 + b * x + c


parameters, covariance = curve_fit(model, users, response_ms)
a, b, c = parameters
predicted = model(users, a, b, c)
rmse = np.sqrt(np.mean((response_ms - predicted) ** 2))

print(f"Fitted model: y = {a:.6f}x^2 + {b:.4f}x + {c:.2f}")
print(f"RMSE: {rmse:.2f} ms")
print(f"Predicted response at 400 users: {model(400, a, b, c):.2f} ms")

os.makedirs("results", exist_ok=True)
x_plot = np.linspace(users.min(), 400, 300)
plt.scatter(users, response_ms, label="Measurements")
plt.plot(x_plot, model(x_plot, a, b, c), label="Fitted curve")
plt.xlabel("Concurrent users")
plt.ylabel("Response time (ms)")
plt.title("Server response-time curve fitting")
plt.legend()
plt.grid(True)
plt.savefig("results/task_14.png", dpi=150)
plt.show()

# ---------- Changed parameters: other models and an altered data point ----------
for deg, name in ((1, "linear"), (3, "cubic")):
    coef = np.polyfit(users, response_ms, deg)
    r = np.sqrt(np.mean((response_ms - np.polyval(coef, users)) ** 2))
    print(f"\n[Changed] {name}: RMSE {r:.2f}, at 400 users {np.polyval(coef, 400):.1f} ms")
altered = response_ms.copy()
altered[-1] = 350
p2, _ = curve_fit(model, users, altered)
print("[Changed] last point 350 -> prediction at 400:", model(400, *p2))
print("Residuals:", np.round(response_ms - predicted, 2))

# ---------- Modified case: exponential growth of app users (thousands) ----------
t = np.arange(6.0)
usr = np.array([100, 135, 182, 246, 332, 448.0])
growth = lambda t, a, k: a * np.exp(k * t)
p, _ = curve_fit(growth, t, usr, p0=[100, 0.3])
print(f"\n[Modified] a={p[0]:.2f}, k={p[1]:.4f}, year 6 forecast {growth(6, *p):.0f}k")
