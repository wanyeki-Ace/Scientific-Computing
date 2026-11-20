# BIT 2118 Scientific Computing: Group Work (Tasks 2 to 20)

Student: Grace Wanjiru Wanyeki, BSCCS/2025/39629
Course: BIT 2118 Scientific Computing, Mount Kenya University

This is my individual repository for the group work. The discussion was done as a group; the code, changed parameters and modified real-world cases here are my own submission.

## Contents

- code/ : one Python program per task (Task_02 to Task_20). Each file has three parts: the program from the brief, a changed parameter, and a modified real-world case, with the assumptions written in comments.
- docs/BSCCS202539629_BIT2118_GroupWork.md : full documentation for every task: inputs and operation, effect of the changed parameter, interpretation of the output and graph, and the modified case with assumptions, data, output and interpretation.
- results/ : graphs saved by the programs that plot (Tasks 8, 10, 13, 14, 15, 17, 18, 20).
- requirements.txt : packages needed (NumPy 2.0 or newer, because Task 16 uses np.trapezoid).

## Tasks

| Task | Topic | File |
|---|---|---|
| 2 | NumPy arrays and numerical computation | code/Task_02_numpy_arrays.py |
| 3 | Mathematical functions and vectorization | code/Task_03_vectorization.py |
| 4 | Floating-point arithmetic and numerical errors | code/Task_04_numerical_errors.py |
| 5 | Matrix algebra and matrix operations | code/Task_05_matrix_algebra.py |
| 6 | Solution of systems of linear equations | code/Task_06_linear_systems.py |
| 7 | Gaussian elimination and LU decomposition | code/Task_07_gaussian_lu.py |
| 8 | Iterative methods for linear systems | code/Task_08_iterative_linear.py |
| 9 | Eigenvalues and eigenvectors | code/Task_09_eigenvalues.py |
| 10 | Root finding: bisection and fixed point | code/Task_10_bisection_fixed_point.py |
| 11 | Root finding: Newton-Raphson and secant | code/Task_11_newton_secant.py |
| 12 | Polynomial and nonlinear equation solving | code/Task_12_nonlinear_equations.py |
| 13 | Interpolation | code/Task_13_interpolation.py |
| 14 | Curve fitting and approximation | code/Task_14_curve_fitting.py |
| 15 | Numerical differentiation | code/Task_15_numerical_differentiation.py |
| 16 | Numerical integration | code/Task_16_numerical_integration.py |
| 17 | ODEs: Euler method | code/Task_17_euler_ode.py |
| 18 | Numerical solution of ODEs using SciPy | code/Task_18_scipy_ode.py |
| 19 | Numerical optimization | code/Task_19_optimization.py |
| 20 | Random numbers and Monte Carlo methods | code/Task_20_monte_carlo.py |

## How to run

```
pip install -r requirements.txt
python code/Task_02_numpy_arrays.py
```

Run from the repository root so the graphs are saved into results/.

## Modified real-world cases

| Task | Modified case |
|---|---|
| 2 | Hourly rainfall at a weather station |
| 3 | Gamma correction of a 1000 x 1000 image |
| 4 | Accumulated error when adding a price 0.1 ten times |
| 5 | Rotating and scaling robot-arm waypoints |
| 6 | Mesh currents in a two-loop circuit |
| 7 | Heating each of three coupled rooms |
| 8 | Steady temperature along a metal rod |
| 9 | Star network importance and a PCA example |
| 10 | Business break-even points |
| 11 | IRR of a solar-installation project |
| 12 | Market equilibrium (supply meets demand) |
| 13 | River gauge with a missing day |
| 14 | Exponential growth of app users |
| 15 | Drone climb rate from altitude samples |
| 16 | Distance from velocity data and energy cost |
| 17 | Logistic population growth |
| 18 | SIR epidemic model |
| 19 | Cheapest animal-feed mix |
| 20 | Project completion risk |
