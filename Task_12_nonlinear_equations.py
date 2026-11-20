import numpy as np
from scipy.optimize import root


# ---------- Brief program: circle x^2 + y^2 = 4 and parabola y = x^2 - 1 ----------
def equations(z):
    x, y = z
    return [x ** 2 + y ** 2 - 4, y - (x ** 2 - 1)]


for guess in ([1.0, 1.0], [-1.0, 1.0]):
    solution = root(equations, guess)
    print("Initial guess:", guess)
    print("Intersection:", solution.x)
    print("Residuals:", equations(solution.x))
    print("Converged:", solution.success)
    print()

# ---------- Changed parameters: bad guesses and a larger circle ----------
for guess in ([0.0, -2.0], [0.0, 0.0]):
    s = root(equations, guess)
    print(f"[Changed] guess {guess}: success={s.success}, residuals={equations(s.x)}")


def equations_r3(z):
    x, y = z
    return [x ** 2 + y ** 2 - 9, y - (x ** 2 - 1)]


for guess in ([1.0, 1.0], [-1.0, 1.0]):
    print("[Changed] radius 3:", root(equations_r3, guess).x)

print("Roots of x^2 - 5x + 6:", np.roots([1, -5, 6]))

# ---------- Modified case: market equilibrium ----------
# Assumptions: supply price 2 + 0.5 q^2, demand price 20 - q (q in thousands)
q_roots = np.roots([0.5, 1, -18])
q = q_roots[q_roots > 0][0]
print(f"\n[Modified] equilibrium quantity {q:.3f}, price {20 - q:.2f}")
