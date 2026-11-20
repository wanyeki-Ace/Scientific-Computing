import numpy as np
from scipy.optimize import minimize


# ---------- Brief program ----------
# Cost per unit: A = 12, B = 18; capacity per unit: A = 100, B = 170; need >= 1000
def cost(z):
    x, y = z
    return 12 * x + 18 * y


def capacity_constraint(z):
    x, y = z
    return 100 * x + 170 * y - 1000


constraints = {"type": "ineq", "fun": capacity_constraint}
bounds = [(0, None), (0, None)]

solution = minimize(cost, x0=[5, 3], method="SLSQP", bounds=bounds, constraints=constraints)
x, y = solution.x

print("Optimization success:", solution.success)
print(f"Type A allocation: {x:.3f}")
print(f"Type B allocation: {y:.3f}")
print(f"Total capacity: {100*x + 170*y:.2f}")
print(f"Minimum planning cost: {solution.fun:.2f}")

best_integer = None
for a in range(0, 20):
    for b in range(0, 20):
        capacity = 100 * a + 170 * b
        if capacity >= 1000:
            total_cost = 12 * a + 18 * b
            if best_integer is None or total_cost < best_integer[0]:
                best_integer = (total_cost, a, b, capacity)
print("Best nearby integer solution (cost, A, B, capacity):", best_integer)

# ---------- Changed parameters: requirement and Type B price ----------
for req in (500, 2000):
    con = {"type": "ineq", "fun": lambda z, req=req: 100 * z[0] + 170 * z[1] - req}
    s = minimize(cost, [5, 3], method="SLSQP", bounds=bounds, constraints=con)
    print(f"\n[Changed] requirement {req}: B={s.x[1]:.2f}, cost {s.fun:.2f}")
cheap_b = lambda z: 12 * z[0] + 14 * z[1]
s = minimize(cheap_b, [5, 3], method="SLSQP", bounds=bounds, constraints=constraints)
print(f"[Changed] Type B cost 14: B={s.x[1]:.2f}, cost {s.fun:.2f}")
print("Capacity per cost: A", round(100 / 12, 2), "B", round(170 / 18, 2))

# ---------- Modified case: cheapest animal-feed mix ----------
# Assumptions: ingredient 1: 30 protein/kg, cost 8; ingredient 2: 12 protein/kg, cost 3; need >= 60
feed_cost = lambda z: 8 * z[0] + 3 * z[1]
protein = {"type": "ineq", "fun": lambda z: 30 * z[0] + 12 * z[1] - 60}
s = minimize(feed_cost, [3, 3], method="SLSQP", bounds=bounds, constraints=protein)
print(f"\n[Modified] feed mix (kg): {np.round(s.x, 3)}, cost {s.fun:.2f}")
