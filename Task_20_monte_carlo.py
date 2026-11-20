import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------- Brief program ----------
rng = np.random.default_rng(42)
capacity = 1000.0
mean_demand = 850.0
std_demand = 120.0
N = 100_000

demand = rng.normal(mean_demand, std_demand, N)
overloaded = demand > capacity
overload_probability = overloaded.mean()

print(f"Estimated overload probability: {overload_probability:.4%}")
print(f"Number of overload cases: {overloaded.sum()} out of {N}")
print(f"Exact value (normal distribution): {1 - norm.cdf(capacity, mean_demand, std_demand):.4%}")

running_probability = np.cumsum(overloaded) / np.arange(1, N + 1)
indices = np.unique(np.logspace(1, np.log10(N), 200).astype(int)) - 1

os.makedirs("results", exist_ok=True)
plt.semilogx(indices + 1, running_probability[indices])
plt.xlabel("Number of simulations")
plt.ylabel("Estimated overload probability")
plt.title("Monte Carlo estimate of server-overload risk")
plt.grid(True)
plt.savefig("results/task_20.png", dpi=150)
plt.show()

# ---------- Changed parameters: N, seed, mean, standard deviation ----------
print()
for n, seed, mu, sd in ((1000, 42, 850, 120), (10_000_000, 42, 850, 120),
                        (100_000, 7, 850, 120), (100_000, 42, 900, 120),
                        (100_000, 42, 850, 150)):
    d = np.random.default_rng(seed).normal(mu, sd, n)
    print(f"[Changed] N={n}, seed={seed}, mean={mu}, sd={sd}: "
          f"{(d > capacity).mean():.4%} (exact {1 - norm.cdf(capacity, mu, sd):.4%})")

rng = np.random.default_rng(42)
px, py = rng.random(N), rng.random(N)
print("Pi estimate:", 4 * np.mean(px ** 2 + py ** 2 <= 1))

# ---------- Modified case: project completion risk ----------
# Assumptions: three sequential tasks, triangular durations (min, likely, max) in days
rng = np.random.default_rng(1)
total = (rng.triangular(5, 8, 14, N) + rng.triangular(3, 4, 9, N)
         + rng.triangular(2, 3, 6, N))
print(f"\n[Modified] mean duration {total.mean():.2f} days, "
      f"P(total > 18 days) = {(total > 18).mean():.2%}")
