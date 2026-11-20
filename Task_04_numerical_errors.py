import numpy as np

# ---------- Brief program ----------
calculated = 0.1 + 0.2
expected = 0.3
print("0.1 + 0.2 =", calculated)
print("Exact comparison:", calculated == expected)
print("Tolerance-based comparison:", np.isclose(calculated, expected))

true_value = np.sqrt(2)
approx_value = 1.414
absolute_error = abs(true_value - approx_value)
relative_error = absolute_error / abs(true_value)
print(f"Absolute error: {absolute_error:.8f}")
print(f"Relative error: {relative_error:.8%}")

for x in [1e-4, 1e-8, 1e-12]:
    direct = (np.sqrt(1 + x) - 1) / x
    stable = 1 / (np.sqrt(1 + x) + 1)
    print(f"x={x:.0e}: direct={direct:.12f}, stable={stable:.12f}")

# ---------- Changed parameter: 1.4142 and smaller x ----------
for approx in (1.414, 1.4142):
    ae = abs(true_value - approx)
    print(f"\n[Changed] approx={approx}: abs={ae:.3e}, rel={ae/true_value:.3e}")
for x in [1e-15, 1e-16]:
    direct = (np.sqrt(1 + x) - 1) / x
    stable = 1 / (np.sqrt(1 + x) + 1)
    print(f"[Changed] x={x:.0e}: direct={direct:.12f}, stable={stable:.12f}")

# ---------- Modified case: adding a 0.1 price ten times ----------
total = 0.0
for _ in range(10):
    total += 0.1
print("\n[Modified] total:", total, "== 1.0?", total == 1.0,
      "isclose:", np.isclose(total, 1.0))
