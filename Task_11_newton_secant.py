from scipy.optimize import root_scalar

# ---------- Brief program: internal rate of return ----------
cash_flows = [-10000, 3000, 3500, 4000, 2500]


def npv_of(cf):
    return lambda rate: sum(c / (1 + rate) ** t for t, c in enumerate(cf))


def d_npv_of(cf):
    return lambda rate: sum(-t * c / (1 + rate) ** (t + 1)
                            for t, c in enumerate(cf) if t > 0)


npv = npv_of(cash_flows)
d_npv = d_npv_of(cash_flows)

newton = root_scalar(npv, fprime=d_npv, x0=0.10, method="newton", xtol=1e-12)
secant = root_scalar(npv, x0=0.05, x1=0.20, method="secant", xtol=1e-12)

print(f"IRR using Newton-Raphson: {newton.root:.6%} ({newton.iterations} iterations)")
print(f"IRR using Secant method: {secant.root:.6%} ({secant.iterations} iterations)")
print(f"NPV at Newton solution: {npv(newton.root):.6f}")

# ---------- Changed parameters: starting guess and hurdle rate ----------
for x0 in (0.5, 0.01):
    r = root_scalar(npv, fprime=d_npv, x0=x0, method="newton")
    print(f"\n[Changed] Newton start {x0}: IRR {r.root:.6%}")
print("[Changed] NPV at a 12% required return:", npv(0.12))

# ---------- Modified case: larger solar-installation project ----------
cf2 = [-15000, 4000, 4500, 5000, 5500, 3000]
r = root_scalar(npv_of(cf2), fprime=d_npv_of(cf2), x0=0.10, method="newton", xtol=1e-12)
print(f"\n[Modified] IRR: {r.root:.4%} ({r.iterations} iterations)")

# Hand-check of Newton on f(x) = x^2 - 2 starting at 1.5
x = 1.5
for _ in range(4):
    x = x - (x * x - 2) / (2 * x)
    print("Newton sqrt(2) step:", x)
