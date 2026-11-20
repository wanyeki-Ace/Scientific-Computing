import numpy as np

# ---------- Brief program ----------
# x = number of Type A servers, y = number of Type B servers
A = np.array([
    [4.0, 2.0],
    [8.0, 12.0]
])
b = np.array([20.0, 72.0])

solution = np.linalg.solve(A, b)
residual = b - A @ solution

print(f"Type A servers: {solution[0]:.0f}")
print(f"Type B servers: {solution[1]:.0f}")
print("Verification A @ solution:", A @ solution)
print("Residual:", residual)
print("Residual norm:", np.linalg.norm(residual))

# ---------- Changed parameter: totals 30 CPU and 100 GB ----------
print("\n[Changed] solution:", np.linalg.solve(A, np.array([30.0, 100.0])))

# ---------- Modified case: two-loop circuit (mesh currents) ----------
# 10 I1 - 4 I2 = 12 ; -4 I1 + 12 I2 = 6
Ac = np.array([[10.0, -4.0], [-4.0, 12.0]])
V = np.array([12.0, 6.0])
I = np.linalg.solve(Ac, V)
print("\n[Modified] loop currents (A):", I)
print("[Modified] check A @ I:", Ac @ I)
