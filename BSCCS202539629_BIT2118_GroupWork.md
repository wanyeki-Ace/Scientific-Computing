# BIT 2118 Scientific Computing: Group Work Guide (Tasks 2 to 20)

Student: Grace Wanjiru Wanyeki, BSCCS/2025/39629

All numbers below come from actually running the programs in the brief (Python 3, NumPy 2.x, SciPy, seeded random generators). Timing figures depend on the machine, so yours will differ.

Prerequisite: Data Science Essentials with Python (self-paced) is not covered here; complete it before the presentation tasks.

How to use this guide. For every task it follows the brief's instructions in order:

- B: inputs and the mathematical operation
- C: change one parameter and explain the effect
- D: meaning of the output (and graph, where there is one)
- E: a modified, related real-world case with assumptions, data, output and interpretation (copy into your own GitHub repository, not a group one)

Notes on the brief's code:

- Task 16 uses np.trapezoid, which needs NumPy 2.0 or newer. On older NumPy use np.trapz.
- The Task 16 data covers 0 to 12 hours, not a full day, so the energy is for 12 hours.
- The task numbers in the brief table (2 to 20) match the sections A to S in order.

---

## Method summary tables

### Where each task sits

| Task | Section | Topic | Key function |
|---|---|---|---|
| 2 | A | Arrays | np.array, np.mean |
| 3 | B | Vectorization | array expressions, timeit |
| 4 | C | Floating-point error | np.isclose |
| 5 | D | Matrix algebra | @, np.linalg.det |
| 6 | E | Linear systems | np.linalg.solve |
| 7 | F | LU decomposition | lu_factor, lu_solve |
| 8 | G | Iterative linear methods | Jacobi, Gauss-Seidel |
| 9 | H | Eigenvalues | np.linalg.eig |
| 10 | I | Bisection, fixed point | root_scalar(bisect) |
| 11 | J | Newton, secant | root_scalar(newton, secant) |
| 12 | K | Nonlinear systems | scipy.optimize.root |
| 13 | L | Interpolation | interp1d, CubicSpline |
| 14 | M | Curve fitting | curve_fit |
| 15 | N | Differentiation | finite differences |
| 16 | O | Integration | trapezoid, simpson |
| 17 | P | Euler ODE | manual loop |
| 18 | Q | SciPy ODE | solve_ivp |
| 19 | R | Optimization | minimize |
| 20 | S | Monte Carlo | rng.normal |

### Root-finding methods compared (Tasks 10 and 11)

| Method | Needs | Speed | Safe? | Result in this guide |
|---|---|---|---|---|
| Bisection | Sign-change bracket | Slow, halves interval | Always converges | 38 iterations, tol 1e-10 |
| Fixed point | Rewrite x = g(x) with abs(g') < 1 | Moderate | Only if abs(g') < 1 | 13 iterations |
| Newton-Raphson | Derivative | Very fast | Needs a good start | 4 iterations (IRR) |
| Secant | Two starting values | Fast | Needs a good start | 7 iterations (IRR) |

### Direct versus iterative linear solvers (Tasks 6, 7, 8)

| Method | Best for | Cost idea | Fails when |
|---|---|---|---|
| np.linalg.solve | One small or medium system | Full elimination each time | Matrix singular |
| LU (lu_factor, lu_solve) | Same A, many b vectors | Factor once, reuse | Matrix singular |
| Jacobi | Large sparse, diagonally dominant | Many cheap sweeps | Not diagonally dominant |
| Gauss-Seidel | Same, about twice as fast as Jacobi | Uses newest values immediately | Not diagonally dominant |

---

## Task 2 (Section A): NumPy Arrays and Numerical Computation

Mnemonic: A-M-M-F = Array, Mean, Min/Max, Fahrenheit-by-formula (one expression for every element).

B. Inputs and operation

- Input: array of 24 hourly temperatures in C (17.5 to 29.4).
- Operations: mean = sum / n, min, max, and F = 1.8C + 32 applied to every element at once (vectorized).
- Output: mean 23.12 C, min 17.50 C, max 29.40 C, first six Fahrenheit values [65.3, 64.4, 64.04, 63.5, 64.22, 66.2].

C. Parameter change: add 5 C to every reading (a hotter day).

- Mean 23.12 becomes 28.12, min 22.5, max 34.4, first Fahrenheit value 74.3.
- Every statistic of location (mean, min, max) shifts by exactly 5 C, while the spread (max minus min = 11.9) is unchanged. In Fahrenheit the shift is 5 x 1.8 = 9 F.

D. Meaning of the output: the day averaged about 23 C, was coolest (17.5 C) around 03:00 and hottest (29.4 C) at midday. The Fahrenheit array shows the same shape in another unit.

E. Modified case: hourly rainfall at a weather station (mm), converted to inches.

- Assumptions: 24 readings from 00:00 to 23:00; most hours dry; inches = mm / 25.4.
- Input: [0,0,0,0,0,0,0.2,0.5,0,0,0,1.2,3.4,5.0,2.1,0,0,0,0,0,0,0,0,0]

```python
import numpy as np
rain_mm = np.array([0,0,0,0,0,0,0.2,0.5,0,0,0,1.2,3.4,5.0,2.1,0,0,0,0,0,0,0,0,0.0])
print("Total:", rain_mm.sum(), "mm")
print("Peak:", rain_mm.max(), "mm at hour", rain_mm.argmax())
print("Rainy hours:", (rain_mm > 0).sum())
rain_in = rain_mm / 25.4
```

- Output: total 12.4 mm, peak 5.0 mm at 13:00, 6 rainy hours. The peak hour in inches is 0.197.
- Interpretation: a short afternoon storm delivered most of the rain; the mean would hide that, so max and argmax are the useful statistics here.

---

## Task 3 (Section B): Mathematical Functions and Vectorization

Mnemonic: "No loop, no wait": NumPy loops in C, so write the expression once and let it run on the whole array.

B. Inputs and operation

- Input: 1,000,000 simulated latencies, uniform between 5 and 250 ms (seed 42).
- Operation: seconds = ms / 1000, then square (the error term). Done once with a Python for loop and once as a vectorized expression, then timed with timeit.
- Output: first five squared values [0.0379, 0.0127, 0.0464, 0.0309, 0.00079] s squared.

C. Parameter change: reduce N from 1,000,000 to 100,000.

| N | Loop time | Vectorized time | Speed-up |
|---|---|---|---|
| 1,000,000 | 0.173 s | 0.0028 s | about 62x |
| 100,000 | 0.0155 s | 0.00012 s | about 130x |

Both times scale roughly with N, but the speed-up ratio is not fixed: timing noise and cache effects change it from run to run. The robust conclusion is that vectorized code is one to two orders of magnitude faster.

D. Meaning: the two methods give identical numbers; only the running time differs. The speed-up shows why large simulations, images and machine learning use arrays.

E. Modified case: brightness correction of a 1000 x 1000 grayscale image (gamma correction).

- Assumptions: pixel values 0 to 255 (random test image), gamma 0.5 brightens dark areas: out = 255 x (pixel/255)^0.5.

```python
import numpy as np
rng = np.random.default_rng(42)
img = rng.integers(0, 256, (1000, 1000)).astype(float)
bright = 255 * (img / 255) ** 0.5
print(img.mean(), bright.mean())
```

- Output: mean pixel value 127.59 becomes 169.91.
- Interpretation: one vectorized line transformed a million pixels; the image becomes brighter overall because values below 255 are raised by the square root.

---

## Task 4 (Section C): Floating-Point Arithmetic and Numerical Errors

Mnemonic: "Never == on floats; use isclose": Absolute = |true - approx|; Relative = Absolute / |true|.

B. Inputs and operation

- Inputs: 0.1 + 0.2 versus 0.3; true value sqrt(2) versus approximation 1.414; the expression (sqrt(1+x) - 1)/x for small x.
- Operations: exact comparison, np.isclose, absolute and relative error, and a stable algebraic rewrite 1/(sqrt(1+x)+1).
- Output: 0.1 + 0.2 = 0.30000000000000004; == gives False, isclose gives True; absolute error 0.00021356, relative error 0.0151%.

C. Parameter change: use 1.4142 instead of 1.414.

- Absolute error drops from 2.1e-4 to 1.36e-5, relative error from 1.51e-4 to 9.6e-6 (about ten times smaller for one more digit).
- For the unstable formula, as x shrinks the direct version breaks down while the stable one stays correct:

| x | direct | stable |
|---|---|---|
| 1e-4 | 0.4999875006 | 0.4999875006 |
| 1e-8 | 0.4999999970 | 0.4999999988 |
| 1e-12 | 0.5000444503 | 0.5000000000 |
| 1e-15 | 0.4440892099 | 0.5000000000 |
| 1e-16 | 0.0 | 0.5 |

D. Meaning: computers store numbers in binary, so 0.1 and 0.2 are stored approximately. Subtracting nearly equal numbers (sqrt(1+x) - 1 for small x) cancels the correct digits and leaves round-off noise (catastrophic cancellation). The mathematically equal stable form avoids the subtraction.

E. Modified case: accumulated error when adding a price of 0.1 many times (a banking-style total).

```python
import numpy as np
total = 0.0
for _ in range(10):
    total += 0.1
print(total, total == 1.0, np.isclose(total, 1.0))
```

- Output: 0.9999999999999999, exact equality False, isclose True.
- Interpretation: ten additions of 0.1 do not reach exactly 1.0. In banking software, use integer cents or the decimal module, and compare with a tolerance.

---

## Task 5 (Section D): Matrix Algebra and Matrix Operations

Mnemonic: "Rotate then Scale = R times S, applied right to left": the matrix next to the point acts first. For row-vector points use points @ M.T.

B. Inputs and operation

- Input: unit square with corners (0,0), (1,0), (1,1), (0,1); rotation 45 degrees; scaling 2 in x and 1.5 in y.
- Operation: M = R @ S, then points @ M.T. Determinant of M = det(R) x det(S) = 1 x 3.
- Output: transformed corners (0,0), (1.4142,1.4142), (0.3536,2.4749), (-1.0607,1.0607); determinant 3.0.

C. Parameter change: rotate 90 degrees instead of 45.

- The corners become (0,0), (0,2), (-1.5,2), (-1.5,0) (up to rounding noise around 1e-16). The shape is a 2 x 1.5 rectangle rotated a quarter turn; the determinant remains 3.0 because rotation does not change area.
- Setting a scale factor to 0 makes the determinant 0 (singular matrix): the square collapses into a line and the transformation cannot be reversed.

D. Meaning: determinant 3 means the area of the square is multiplied by 3 (2 x 1.5). The sign (positive) means no mirror reflection.

E. Modified case: rescaling and rotating a robot-arm waypoint path.

- Assumptions: waypoints in metres, rotated 30 degrees to align with a new workstation and shrunk to 0.8 scale.

```python
import numpy as np
path = np.array([[0,0],[2,0],[2,1],[0,1.]])
a = np.deg2rad(30)
R = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
S = 0.8 * np.eye(2)
M = R @ S
new_path = path @ M.T
print(new_path, np.linalg.det(M))   # 0.64
```

- Interpretation: determinant 0.64 equals 0.8 squared, confirming the area shrinks by 36 percent.

---

## Task 6 (Section E): Solution of Systems of Linear Equations

Mnemonic: A x = b, "A times unknowns equals results": solve, then verify with A @ x and the residual.

B. Inputs and operation

- Inputs: A = [[4,2],[8,12]] (CPU and memory per server), b = [20, 72] (observed totals).
- Operation: x = A^-1 b computed by np.linalg.solve (LU based, more accurate than inverting).
- Output: Type A servers 3, Type B servers 4; residual 0.

C. Parameter change: totals become 30 CPU and 100 GB.

- Solution becomes 5 and 5. Check: 4(5) + 2(5) = 30 and 8(5) + 12(5) = 100.
- Interpretation: more total resource use means more active servers; a change in b moves x linearly.

D. Meaning: the residual (b minus A x) near 0 proves the answer reproduces the data. A non-whole-number result would signal inconsistent measurements.

E. Modified case: loop currents in an electrical circuit (mesh analysis).

- Assumptions: two loops sharing a 4 ohm resistor; loop 1 has 10 ohm total and a 12 V source, loop 2 has 12 ohm total and a 6 V source.
- Equations: 10 I1 - 4 I2 = 12 and -4 I1 + 12 I2 = 6.

```python
import numpy as np
A = np.array([[10., -4.], [-4., 12.]])
V = np.array([12., 6.])
I = np.linalg.solve(A, V)
print(I, A @ I)
```

- Output: I1 = 1.615 A, I2 = 1.038 A, and A @ I returns [12, 6].
- Interpretation: both currents are positive, so both flow in the assumed directions.

---

## Task 7 (Section F): Gaussian Elimination and LU Decomposition

Mnemonic: "Factor once, solve many": PA = LU; Lower and Upper triangles.

B. Inputs and operation

- Inputs: 3 x 3 matrix A (diagonal 4, 4, 3, off-diagonals -1) and three measurement vectors b.
- Operation: lu_factor(A) once, then lu_solve for each b.
- Output: solutions [5, 5, 5], [5.317, 5.268, 4.756], [4.902, 5.610, 5.537]; residual norms around 1e-15 or exactly 0; the direct solution for the first set matches [5, 5, 5]. Condition number 2.39 (well conditioned).

C. Parameter change: new measurement vector b = [20, 0, 0].

- Solution [5.366, 1.463, 0.488]: all load at the first node, so the response decays with distance from it. No new factorization was needed.
- Timing check for a 1500 x 1500 matrix with 20 right-hand sides: repeated np.linalg.solve about 1.05 s versus LU factor-once about 0.074 s (roughly 14x on this machine).

D. Meaning: tiny residuals confirm accuracy. The speed advantage appears when A stays fixed and only b changes, as in a simulation stepping through time.

E. Modified case: steady heat flow in three connected rooms with different heat sources.

- Assumptions: matrix A describes coupling between rooms (same as brief); b are heater outputs per room.

```python
from scipy.linalg import lu_factor, lu_solve
import numpy as np
A = np.array([[4,-1,0],[-1,4,-1],[0,-1,3.]])
lu, piv = lu_factor(A)
for b in ([20,0,0],[0,20,0],[0,0,20]):
    print(lu_solve((lu, piv), np.array(b, float)))
```

- Interpretation: each run shows how heating one room spreads to the others; one factorization served all three scenarios.

---

## Task 8 (Section G): Iterative Methods for Linear Systems

Mnemonic: "Guess, Update, Check": Jacobi uses old values (J = Just the old), Gauss-Seidel uses the newest values (GS = Gets the Spot instantly), so it is about twice as fast.

B. Inputs and operation

- Inputs: 5 x 5 tridiagonal matrix (2 on the diagonal, -1 beside it), b = [100,0,0,0,20] (boundary temperatures 100 C and 20 C), tolerance 1e-8, initial guess zeros.
- Operation: repeat x_new = (b - R x) / D (Jacobi) or update one unknown at a time (Gauss-Seidel) until the largest change is below tolerance.
- Output: both reach about [86.667, 73.333, 60, 46.667, 33.333], equal to the direct solution. Jacobi needed 150 iterations, Gauss-Seidel 75.

C. Parameter changes

| Change | Jacobi iterations | Gauss-Seidel iterations |
|---|---|---|
| tol = 1e-4 | 86 | 43 |
| tol = 1e-8 (brief) | 150 | 75 |
| tol = 1e-12 | 214 | 107 |
| Left boundary 150 C (tol 1e-8) | 152 | 76 |

Tighter tolerance costs more iterations, but each extra digit costs a roughly constant number of sweeps. Changing the left boundary to 150 C gives [128.33, 106.67, 85, 63.33, 41.67] with almost the same iteration count, because the convergence rate depends on the matrix, not on b.

Failure example: for [[1,2],[3,1]] (not diagonally dominant) Jacobi diverges to values near 1e19 after 50 iterations.

D. Graph interpretation: the semilog plot of the maximum change per iteration is a straight line going down, meaning the error shrinks by a constant factor every iteration (linear convergence). The Gauss-Seidel line is steeper (about twice the slope).

E. Modified case: steady temperature along a metal rod with 7 interior points, ends held at 80 C and 30 C.

```python
import numpy as np
n = 7
A = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
b = np.zeros(n); b[0] = 80; b[-1] = 30
print(np.linalg.solve(A, b))
```

- Output: [73.75, 67.5, 61.25, 55, 48.75, 42.5, 36.25]; Jacobi 263 iterations, Gauss-Seidel 132.
- Interpretation: the temperature falls linearly by 6.25 C per node, which is the correct steady-state conduction profile. More grid points means more iterations, which is why preconditioning matters for large grids.

---

## Task 9 (Section H): Eigenvalues and Eigenvectors

Mnemonic: A v = lambda v, "A leaves its direction alone": the dominant eigenvector of the adjacency matrix gives influence (eigenvector centrality).

B. Inputs and operation

- Input: adjacency matrix of a 4-node undirected network (edges A-B, A-C, B-C, B-D, C-D).
- Operation: np.linalg.eig, pick the largest real eigenvalue, take absolute values of its eigenvector and normalize so the scores sum to 1.
- Output: dominant eigenvalue 2.5616, which is (1 + sqrt(17))/2. Importance: A 0.2192, B 0.2808, C 0.2808, D 0.2192.

C. Parameter change: add the missing edge A-D.

- The network becomes complete (every node has 3 links), the eigenvalue becomes 3, and every node scores 0.25. When all nodes are structurally identical, none is more important.

D. Meaning: B and C connect to all other nodes, so they score highest. A and D are equal by symmetry. The eigenvalue measures the overall connectivity; the verification error ||Av - lambda v|| is near 0.

E. Modified cases

1. Star network of 5 nodes (a hub with 4 leaves): dominant eigenvalue 2 (square root of 4), hub score 0.3333, each leaf 0.1667. In a real network the hub is the single point of failure or the best place for a monitor.
2. PCA idea: for the covariance matrix of two correlated variables (200 synthetic samples, seed 0), the eigenvalue ratio was 0.917 to 0.083. The first principal component carries 91.7 percent of the variance, so one dimension is almost enough.

```python
import numpy as np
S = np.zeros((5,5)); S[0,1:] = 1; S[1:,0] = 1
w, v = np.linalg.eigh(S)
score = np.abs(v[:, -1]); print(w[-1], score / score.sum())
```

---

## Task 10 (Section I): Root Finding, Bisection and Fixed-Point Methods

Mnemonic: Bisection = "Bracket, Halve, Keep the sign change". Fixed point = "Rearrange x = g(x), repeat, need slope under 1".

B. Inputs and operation

- Input: f(T) = T - 25 - 10 exp(-T/20), bracket [20, 40] where f(20) is about -8.7 and f(40) about +13.7 (opposite signs). Fixed-point form g(T) = 25 + 10 exp(-T/20), start T = 30.
- Output: equilibrium temperature 27.525211 C by both methods; bisection took 38 iterations (tol 1e-10), fixed point 13 iterations.

C. Parameter change

| Change | Bisection iterations |
|---|---|
| xtol 1e-3 | 15 |
| xtol 1e-10 (brief) | 38 |
| xtol 1e-12 | 45 |
| Wider bracket [0, 100], xtol 1e-10 | 40 |

Bisection halves the interval each time, so one more decimal digit costs about 3.3 more iterations. A wider bracket adds only two iterations. Fixed point converges faster here because |g'(T)| = 0.5 exp(-T/20) is about 0.13, much smaller than 1.

D. Graph interpretation: the curve of f(T) crosses the horizontal zero line once; the dashed vertical line sits at 27.5 C. To the left f is negative (loss exceeds gain), to the right positive.

E. Modified case: break-even points of a business.

- Assumptions: revenue 45 q; cost 12000 + 20 q + 0.002 q squared (units q).
- Profit(q) = -0.002 q squared + 25 q - 12000, which has two roots.

```python
from scipy.optimize import root_scalar
profit = lambda q: 45*q - (12000 + 20*q + 0.002*q**2)
print(root_scalar(profit, bracket=[100, 1000], method="bisect").root)
print(root_scalar(profit, bracket=[5000, 12000], method="bisect").root)
```

- Output: 500 and 12,000 units. Profit is positive only between them: below 500 fixed costs dominate, above 12,000 the quadratic cost term overtakes revenue. Note that two brackets are needed because a function with two roots has two sign changes.

---

## Task 11 (Section J): Root Finding, Newton-Raphson and Secant Methods

Mnemonic: Newton = "Next = Now minus f over f-prime" (tangent line). Secant = "Skip the derivative, Slope from two points".

B. Inputs and operation

- Inputs: cash flows [-10000, 3000, 3500, 4000, 2500]; NPV(r) = sum of cash_t / (1+r)^t; Newton start 0.10; secant starts 0.05 and 0.20.
- Operation: find r with NPV(r) = 0 (the internal rate of return, IRR).
- Output: IRR 11.5425 percent by both methods; Newton 4 iterations, secant 7; NPV at the answer about 2e-12.

C. Parameter change

- Newton starting at 0.5 or 0.01 gives the same IRR (11.5425 percent): this NPV curve is smooth and decreasing, so the start is forgiving.
- Evaluating NPV at a required return of 12 percent gives -95.33: the project fails at a 12 percent hurdle rate because 12 percent is above the IRR.

D. Meaning: an IRR of 11.54 percent is the yearly return the project earns on the money invested. If the cost of capital is below it the project adds value.

Hand check (Newton on x squared - 2, x0 = 1.5): 1.41667, 1.414216, 1.41421356237, 1.4142135623731. The number of correct digits roughly doubles each step (quadratic convergence), which is why Newton took only 4 steps.

E. Modified case: IRR for a larger solar-installation loan project.

- Assumptions: invest 15,000, then yearly inflows 4000, 4500, 5000, 5500, 3000.

```python
from scipy.optimize import root_scalar
cf = [-15000, 4000, 4500, 5000, 5500, 3000]
npv = lambda r: sum(c/(1+r)**t for t, c in enumerate(cf))
dnpv = lambda r: sum(-t*c/(1+r)**(t+1) for t, c in enumerate(cf) if t > 0)
print(root_scalar(npv, fprime=dnpv, x0=0.10, method="newton").root)
```

- Output: IRR 14.4468 percent in 5 Newton iterations. Interpretation: higher than the first project, so it is more attractive if the cost of capital is below 14.4 percent.

---

## Task 12 (Section K): Polynomial and Nonlinear Equation Solving

Mnemonic: "Residual zero, Guess matters": root() drives residuals to zero but only finds the root near your starting guess.

B. Inputs and operation

- Inputs: circle x squared + y squared = 4 and parabola y = x squared - 1; initial guesses [1,1] and [-1,1].
- Operation: scipy.optimize.root finds (x, y) making both residuals zero.
- Output: intersections (1.5175, 1.3028) and (-1.5175, 1.3028), residuals about 1e-13, success True.

C. Parameter change

- Guesses [0,-2] and [0,0] both fail (success False, residual not near 0), because at x = 0 the Jacobian is singular and no step direction exists.
- Circle radius 3 (right side 9): intersections move to (+/-1.8364, 2.3723).

D. Meaning: substituting x squared = y + 1 gives y squared + y - 3 = 0, so y = 1.3028 or y = -2.3028. The second would need x squared = -1.30 (negative), so there are only two real intersections, mirror images in the y axis. This is why two guesses, one each side, are needed.

Polynomial check: np.roots([1, -5, 6]) returns [3, 2] for x squared - 5x + 6.

E. Modified case: market equilibrium where supply meets demand.

- Assumptions: supply price 2 + 0.5 q squared, demand price 20 - q (quantity q in thousands, price in some currency unit).

```python
import numpy as np
print(np.roots([0.5, 1, -18]))   # 0.5 q^2 + q - 18 = 0
```

- Output: q = 5.083 (keep) and q = -7.083 (reject, negative quantity). Equilibrium price = 20 - 5.083 = 14.92.
- Interpretation: only the positive root is meaningful. Always check roots against physical constraints.

---

## Task 13 (Section L): Interpolation

Mnemonic: "Linear = straight lines (L is a line), Cubic = smooth curve". Interpolation passes through every data point; extrapolation beyond them is risky.

B. Inputs and operation

- Inputs: hourly temperatures from 08:00 to 14:00 [18, 20, 22.5, 24, 25, 25.5, 25]; requested every 10 minutes.
- Operation: interp1d (straight lines) and CubicSpline (smooth cubic pieces).
- Output at 10:30: linear 23.25 C, cubic 23.363 C.

C. Parameter change: make the 11:00 reading 30 C (a spike).

- At 10:30 the linear estimate becomes 26.25 C and the cubic 27.006 C. At 11:30 they are 27.5 and 28.19. The cubic curve even rises to 30.03, slightly above the largest reading, showing that splines can overshoot.
- At 08:30 the cubic gives 18.81 versus linear 19.0; the two methods differ most where the curve bends.
- Extrapolating the cubic to 15:00 gives 22.94 C (falling), a guess outside the data; interp1d refuses to extrapolate by default.

D. Graph interpretation: dots are the hourly measurements; the cubic spline curves smoothly, while the linear line has corners at each hour. Both pass through every dot.

E. Modified case: a river-level gauge missed day 3.

- Assumptions: levels in cm on days 0, 1, 2, 4, 5, 6 = 10, 12, 15, 22, 25, 24.

```python
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
d = np.array([0,1,2,4,5,6.]); level = np.array([10,12,15,22,25,24.])
print(float(interp1d(d, level)(3)), float(CubicSpline(d, level)(3)))
```

- Output: 18.5 cm (linear) and 18.41 cm (cubic). The two agree within 0.1 cm, so the missing value can be filled with confidence.

---

## Task 14 (Section M): Curve Fitting and Approximation

Mnemonic: "Minimise Squared Residuals" = least squares. Judge a fit by RMSE and residual pattern, not just the picture.

B. Inputs and operation

- Inputs: users [50 ... 350] and response times [82, 95, 115, 145, 185, 238, 305] ms; model a x squared + b x + c.
- Operation: curve_fit minimizes the sum of squared residuals.
- Output: y = 0.002167 x squared - 0.1345 x + 85.00; RMSE 1.55 ms; prediction at 400 users 377.86 ms.

C. Parameter change

| Model or data change | RMSE (ms) | Prediction at 400 |
|---|---|---|
| Quadratic (brief) | 1.55 | 377.9 ms |
| Linear | 18.83 | 312.9 ms |
| Cubic (4 parameters, 7 points) | 0.14 | 387.9 ms |
| Quadratic, last point changed 305 to 350 | n/a | 435.7 ms |

A straight line fits poorly (RMSE 18.8). The cubic fits almost perfectly but with 4 parameters for 7 points it risks overfitting. One altered end point moves the prediction at 400 users from 378 to 436 ms, so extrapolation is sensitive to the last data points.

D. Meaning: the response time grows faster than linearly as load rises, a sign of queuing and saturation. The residuals (about -1.8 to +1.8 ms) are small and mixed in sign, so the quadratic captures the main pattern, although the signs show a mild wave that a few more points would help to confirm.

E. Modified case: exponential growth of app users.

- Assumptions: users in thousands for years 0 to 5 are [100, 135, 182, 246, 332, 448]; model a exp(k t).

```python
import numpy as np
from scipy.optimize import curve_fit
t = np.arange(6.); users = np.array([100,135,182,246,332,448.])
f = lambda t, a, k: a*np.exp(k*t)
p, _ = curve_fit(f, t, users, p0=[100, 0.3])
print(p, f(6, *p))
```

- Output: a = 99.99, k = 0.29997 (about 30 percent continuous growth per year), prediction for year 6 about 605 thousand, RMSE 0.09.
- Interpretation: growth cannot stay exponential forever; use the forecast only for the near term.

---

## Task 15 (Section N): Numerical Differentiation

Mnemonic: "Central is Centred, Better by one order": forward error is proportional to h, central error to h squared. Too small an h brings round-off back.

B. Inputs and operation

- Inputs: GPS time 0 to 6 s and positions [0, 4.8, 10.2, 16.1, 22.7, 30, 38] m.
- Operation: forward difference at the first point, central differences inside, backward difference at the last.
- Output: speeds [4.8, 5.1, 5.65, 6.25, 6.95, 7.65, 8.0] m/s, identical to np.gradient.

C. Parameter changes

1. Change the position at t = 3 to 17.5 (a GPS glitch): the speeds at t = 2 and t = 4 move to 6.35 and 6.25 instead of 5.65 and 6.95. One bad position contaminates its two neighbors.
2. Step-size test on f(x) = sin x at x = 1:

| h | forward error | central error |
|---|---|---|
| 1 | 0.47 | 0.086 |
| 0.1 | 0.043 | 9.0e-4 |
| 0.01 | 4.2e-3 | 9.0e-6 |
| 1e-5 | 4.2e-6 | 1.1e-11 |
| 1e-8 | 3.0e-9 | 2.6e-9 |
| 1e-12 | 4.3e-5 | 1.2e-5 |

The error first falls (truncation error shrinks) and then rises again below about 1e-8 (round-off error from Task 4 takes over).

D. Graph interpretation: the vehicle speed rises steadily from 4.8 to 8 m/s, so it is accelerating (about 0.3 to 0.7 m/s squared). The end values are less reliable because they use one-sided differences.

E. Modified case: climb rate of a drone from altitude samples.

- Assumptions: altitude in metres at 0, 10, 20, 30, 40 s = 0, 120, 260, 430, 640.

```python
import numpy as np
t = np.array([0,10,20,30,40.]); alt = np.array([0,120,260,430,640.])
print(np.gradient(alt, t))
```

- Output: [12, 13, 15.5, 19, 21] m/s. The drone climbs faster as time passes, so it is accelerating.

---

## Task 16 (Section O): Numerical Integration

Mnemonic: "Trapezoid = Two points, Simpson = Squared curve (parabola through three points)". Integral = accumulated total.

B. Inputs and operation

- Inputs: smart-meter power (kW) [0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9] at 0, 2, 4 ... 12 hours.
- Operation: energy = integral of power over time with the trapezoidal rule and Simpson's rule.
- Output: trapezoid 15.700 kWh, Simpson 15.800 kWh, difference 0.100 kWh.

C. Parameter change

- Raise every power value by 20 percent: trapezoid 18.84, Simpson 18.96 kWh. Both scale by exactly 1.2 because integration is linear.
- Test on a known integral, sin x from 0 to pi with only 5 points (exact answer 2): trapezoid 1.896, Simpson 2.0046. Simpson is far more accurate for smooth curves.
- Adding hourly points by linear interpolation: trapezoid 15.700 and Simpson 15.700, so interpolation alone adds no new information.

D. Meaning: the building used about 15.7 to 15.8 kWh in 12 hours. The load peaks at 1.8 kW at 06:00. The 0.1 kWh gap between methods indicates the uncertainty from coarse two-hour sampling.

E. Modified case: distance travelled from velocity data, and cost of the energy.

- Assumptions: velocity in m/s at 0, 2, 4, 6, 8, 10 s = 0, 5, 9, 12, 14, 15; electricity tariff KES 24 per kWh (illustrative).

```python
import numpy as np
from scipy.integrate import simpson
t = np.arange(0, 11, 2.); v = np.array([0,5,9,12,14,15.])
print(np.trapezoid(v, t), simpson(v, x=t))
print(15.8 * 24)   # cost of the 12-hour energy
```

- Output: distance 95.0 m (trapezoid) and 95.83 m (Simpson); the 12-hour energy at KES 24 per kWh costs about KES 379.2.

---

## Task 17 (Section P): Ordinary Differential Equations, Euler Method

Mnemonic: "New = Old + h times slope": y(n+1) = y(n) + h f(t, y). Halve h, halve the error; too large an h blows up.

B. Inputs and operation

- Inputs: ambient 25 C, initial 90 C, cooling constant k = 0.12 per minute, step h = 1 min, end 30 min.
- Operation: Euler update T(n+1) = T(n) + h (-k (T(n) - 25)), compared to the exact T = 25 + 65 exp(-k t).
- Output: Euler 26.40 C, exact 26.78 C, final error 0.372 C.

C. Parameter changes

| h (min) | Euler at 30 min | Final error | Largest error |
|---|---|---|---|
| 5 | 25.27 | 1.51 | 9.67 |
| 2 | 26.06 | 0.717 | 3.20 |
| 1 (brief) | 26.40 | 0.372 | 1.51 |
| 0.5 | 26.59 | 0.189 | 0.74 |
| 0.1 | 26.74 | 0.038 | 0.14 |
| 17 | 95.3 | 69.2 | 76.1 |

The final error is proportional to h (first-order method). With h = 17 min the product k h = 2.04 exceeds 2 and the solution becomes unstable, oscillating and growing instead of cooling.

Other parameters: k = 0.25 gives 25.01 C at 30 min (cools faster); ambient 5 C gives 6.84 C (exact 7.32); starting at 60 C gives 25.76 C. The time to reach 40 C is about 12.2 minutes.

D. Graph interpretation: the temperature decays exponentially, quickly at first (large temperature difference) and slowly later, approaching the ambient value. The Euler markers sit slightly below the exact curve because each step uses the slope at the start of the interval.

E. Modified case: logistic population growth of bacteria.

- Assumptions: dP/dt = r P (1 - P/K), P0 = 1000, r = 0.5 per hour, K = 10,000.

```python
import numpy as np
def euler_logistic(h, P0=1000, r=0.5, K=10000, end=20):
    n = int(end / h); P = P0
    for _ in range(n):
        P += h * r * P * (1 - P / K)
    return P
print(euler_logistic(1), euler_logistic(0.1))   # exact 9995.9
```

- Output: 9999.47 (h = 1) and 9996.45 (h = 0.1) compared with the exact 9995.92. The population rises in an S shape and levels at the carrying capacity near 10,000.

---

## Task 18 (Section Q): Numerical Solution of ODEs Using SciPy

Mnemonic: "solve_ivp = Solve Initial Value Problem": (function, time span, start value). tau = RC is the time constant; at t = tau the capacitor reaches 63 percent.

B. Inputs and operation

- Inputs: Vs = 5 V, R = 1000 ohm, C = 0.001 F (tau = RC = 1 s), V0 = 0, time 0 to 5 s, tolerances rtol 1e-9 and atol 1e-11.
- Operation: solve dV/dt = (Vs - V)/(RC) adaptively and compare with the exact V = Vs (1 - exp(-t/RC)).
- Output: solver success True, voltage at 5 s 4.9663 V, largest error against exact 9.3e-10 V.

C. Parameter changes

| Change | Time constant | V at 5 s | Time to 63 percent |
|---|---|---|---|
| Brief values | 1.0 s | 4.966 V | about 1.0 s |
| C = 4.7 mF | 4.7 s | 3.274 V | about 4.7 s |
| R = 2200 ohm | 2.2 s | 4.485 V | about 2.2 s |
| Vs = 12 V | 1.0 s | 11.919 V | about 1.0 s |
| Discharge from 5 V to 0 | 1.0 s | 0.034 V | about 1.0 s |

Larger R or C slows charging; the supply voltage changes only the final level, not the speed.

D. Graph interpretation: the solid solver curve and the dashed exact curve coincide. The voltage rises steeply then flattens toward 5 V, which is the exponential charging shape.

E. Modified case: SIR epidemic model.

- Assumptions: fractions of the population; recovery rate 0.1 per day; start 99 percent susceptible, 1 percent infected; compare transmission rates 0.3 and 0.15.

```python
import numpy as np
from scipy.integrate import solve_ivp
def sir(t, y, beta, gamma):
    S, I, R = y
    return [-beta*S*I, beta*S*I - gamma*I, gamma*I]
for beta in (0.3, 0.15):
    s = solve_ivp(sir, (0, 160), [0.99, 0.01, 0.0], args=(beta, 0.1),
                  t_eval=np.linspace(0, 160, 1601))
    print(beta, s.y[1].max(), s.t[s.y[1].argmax()], s.y[2, -1])
```

| beta | Basic reproduction number | Peak infected | Day of peak | Ever infected |
|---|---|---|---|---|
| 0.30 | 3.0 | 30.4 percent | 26.7 | 94.2 percent |
| 0.15 | 1.5 | 7.0 percent | 65.2 | 57.8 percent |

- Interpretation: halving transmission (distancing, masks) lowers and delays the peak sharply, easing pressure on hospitals.

---

## Task 19 (Section R): Numerical Optimization

Mnemonic: "Objective, Constraints, Bounds" (O-C-B): what to minimize, what must hold, what limits the variables.

B. Inputs and operation

- Inputs: cost 12 per Type A unit and 18 per Type B; capacity 100 per A and 170 per B; requirement at least 1000; x, y at least 0; start (5, 3).
- Operation: SLSQP minimizes 12x + 18y subject to 100x + 170y - 1000 at least 0.
- Output: Type A 0, Type B 5.882 units, capacity 1000, minimum cost 105.88. Best whole-server solution: 6 Type B, 0 Type A, capacity 1020, cost 108.

C. Parameter changes

| Change | Type B | Minimum cost |
|---|---|---|
| Capacity requirement 500 | 2.94 | 52.94 |
| Requirement 1000 (brief) | 5.88 | 105.88 |
| Requirement 2000 | 11.76 | 211.76 |
| Type B cost reduced to 14 | 5.88 | 82.35 |

Capacity per cost is 8.33 for A versus 9.44 for B, so the optimizer uses only the better-value type and the cost scales in proportion to the requirement. Making B cheaper leaves the choice unchanged but lowers the cost.

D. Meaning: the constraint is exactly active (capacity 1000), so the optimum sits on the boundary. The integer answer costs 2.12 more (108 versus 105.88) because servers cannot be split. Rounding the continuous answer is not always optimal, which is why nearby integers are compared.

E. Modified case: cheapest animal-feed mix.

- Assumptions: ingredient 1 gives 30 protein units per kg and costs 8; ingredient 2 gives 12 protein units per kg and costs 3; need at least 60 protein units.

```python
from scipy.optimize import minimize
cost = lambda z: 8*z[0] + 3*z[1]
con = {"type": "ineq", "fun": lambda z: 30*z[0] + 12*z[1] - 60}
s = minimize(cost, [3, 3], method="SLSQP", bounds=[(0, None)]*2, constraints=con)
print(s.x, s.fun)
```

- Output: 0 kg of ingredient 1, 5 kg of ingredient 2, cost 15.
- Interpretation: ingredient 2 gives 4 protein units per cost unit versus 3.75 for ingredient 1, so it wins on price per protein.

---

## Task 20 (Section S): Random Numbers and Monte Carlo Methods

Mnemonic: "More samples, less noise": the error shrinks like 1 over the square root of N (100 times more samples gives 10 times less error). Use a seed for reproducibility.

B. Inputs and operation

- Inputs: capacity 1000, demand normal with mean 850 and standard deviation 120, N = 100,000, seed 42.
- Operation: generate demand scenarios, test demand greater than capacity, average the True values.
- Output: estimated overload probability 10.616 percent (10,616 of 100,000). The exact normal-distribution value is 10.565 percent.

C. Parameter changes

| Change | Estimate | Exact |
|---|---|---|
| N = 1,000 | 10.1 percent | 10.56 percent |
| N = 100,000 (brief) | 10.616 percent | 10.56 percent |
| N = 10,000,000 | 10.542 percent | 10.56 percent |
| Different seed (7) | 10.558 percent | 10.56 percent |
| Mean demand 900 | 20.22 percent | 20.23 percent |
| Standard deviation 150 | 15.84 percent | 15.87 percent |

Small N is noisy; large N is close to exact. A different seed changes the answer slightly, which reflects sampling noise. Higher mean demand or higher variability raises the risk a lot.

Pi check from the brief's example: 100,000 random points gave 3.1378 (true 3.1416).

D. Graph interpretation: the running probability fluctuates wildly for the first hundred or so simulations, then settles near 0.106. The x-axis is logarithmic so the early noise is visible. This shows convergence of the estimate, not a change in the underlying risk.

E. Modified case: project completion risk with uncertain task durations.

- Assumptions: three sequential tasks with triangular durations (min, most likely, max) in days: (5, 8, 14), (3, 4, 9), (2, 3, 6). Deadline 18 days. N = 100,000, seed 1.

```python
import numpy as np
rng = np.random.default_rng(1)
N = 100_000
total = (rng.triangular(5, 8, 14, N) + rng.triangular(3, 4, 9, N)
         + rng.triangular(2, 3, 6, N))
print((total > 18).mean(), total.mean())
```

- Output: mean duration 17.99 days and a 48.2 percent probability of missing the 18-day deadline.
- Interpretation: the most-likely durations sum to 15, but the right-skewed uncertainty pushes the mean to 18, so an 18-day promise has nearly a coin-flip chance of failing. A buffer (for example 21 days) is needed.

---

## Submission checklist (individual GitHub, per the brief)

- One personal repository, not a group repository.
- Suggested layout: code/ (the brief's programs plus your modified versions), docs/ (this guide split per task or one README), results/ (saved plots).
- For each task, document: inputs and operation, the changed parameter and its effect, the output and graph interpretation, and your modified case with assumptions, data, output and interpretation.
- Submit the repository link and documentation through VLMS. Use letters and numbers only in submission file names, for example BSCCS202539629.
- Save the plots with plt.savefig("results/task_08.png") before plt.show() so the images are in your repository.
