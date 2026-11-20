import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------- Brief program: RC charging, dV/dt = (Vs - V)/(R C) ----------
Vs = 5.0
R = 1000.0
C = 0.001
V0 = 0.0


def rc_circuit(t, V):
    return [(Vs - V[0]) / (R * C)]


t_eval = np.linspace(0, 5, 200)
solution = solve_ivp(rc_circuit, (0, 5), [V0], t_eval=t_eval, rtol=1e-9, atol=1e-11)
exact = Vs * (1 - np.exp(-solution.t / (R * C)))

print("Solver success:", solution.success)
print(f"Capacitor voltage at 5 s: {solution.y[0, -1]:.4f} V")
print(f"Maximum error against exact solution: {np.max(np.abs(solution.y[0]-exact)):.3e} V")

os.makedirs("results", exist_ok=True)
plt.plot(solution.t, solution.y[0], label="solve_ivp")
plt.plot(solution.t, exact, linestyle="--", label="Exact")
plt.xlabel("Time (s)")
plt.ylabel("Capacitor voltage (V)")
plt.title("RC circuit charging")
plt.legend()
plt.grid(True)
plt.savefig("results/task_18.png", dpi=150)
plt.show()


# ---------- Changed parameters: C, R, Vs, discharge ----------
def rc_run(Vs=5.0, R=1000.0, C=0.001, V0=0.0):
    s = solve_ivp(lambda t, V: [(Vs - V[0]) / (R * C)], (0, 5), [V0],
                  t_eval=np.linspace(0, 5, 500), rtol=1e-9, atol=1e-11)
    return s.y[0, -1], R * C


print()
for label, kw in (("C = 4.7 mF", dict(C=0.0047)), ("R = 2200 ohm", dict(R=2200.0)),
                  ("Vs = 12 V", dict(Vs=12.0)), ("discharge from 5 V", dict(V0=5.0, Vs=0.0))):
    v, tau = rc_run(**kw)
    print(f"[Changed] {label}: tau={tau:.2f} s, V(5 s)={v:.3f} V")


# ---------- Modified case: SIR epidemic model ----------
# Assumptions: fractions of population, gamma = 0.1 per day, 1% initially infected
def sir(t, y, beta, gamma):
    S, I, Rr = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]


print()
for beta in (0.3, 0.15):
    s = solve_ivp(sir, (0, 160), [0.99, 0.01, 0.0], args=(beta, 0.1),
                  t_eval=np.linspace(0, 160, 1601))
    print(f"[Modified] beta={beta}: peak infected {s.y[1].max():.3f} on day "
          f"{s.t[s.y[1].argmax()]:.1f}, ever infected {s.y[2, -1]:.3f}")
