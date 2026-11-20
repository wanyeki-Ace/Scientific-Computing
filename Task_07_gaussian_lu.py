import numpy as np
from scipy.linalg import lu_factor, lu_solve

# ---------- Brief program ----------
A = np.array([
    [4.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 3.0]
])

measurement_sets = [
    np.array([15.0, 10.0, 10.0]),
    np.array([16.0, 11.0, 9.0]),
    np.array([14.0, 12.0, 11.0])
]

lu, piv = lu_factor(A)          # factor A only once

for i, b in enumerate(measurement_sets, start=1):
    x = lu_solve((lu, piv), b)
    residual = b - A @ x
    print(f"Measurement set {i}")
    print("Solution:", x)
    print("Residual norm:", np.linalg.norm(residual))
    print()

print("Direct solution for first set:", np.linalg.solve(A, measurement_sets[0]))

# ---------- Changed parameter: new measurement vector ----------
print("\n[Changed] b = [20, 0, 0] ->", lu_solve((lu, piv), np.array([20.0, 0.0, 0.0])))
print("Condition number of A:", np.linalg.cond(A))

# ---------- Modified case: heating each of three rooms in turn ----------
# Assumption: A is the room-coupling matrix, b is the heater output (20 units).
for room, b in zip("123", ([20, 0, 0], [0, 20, 0], [0, 0, 20])):
    print(f"[Modified] heater in room {room}:", lu_solve((lu, piv), np.array(b, float)))
