import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- Brief program ----------
A = np.array([
    [2., -1., 0., 0., 0.],
    [-1., 2., -1., 0., 0.],
    [0., -1., 2., -1., 0.],
    [0., 0., -1., 2., -1.],
    [0., 0., 0., -1., 2.]
])
b = np.array([100., 0., 0., 0., 20.])   # boundaries: 100 C left, 20 C right


def jacobi(A, b, tol=1e-8, max_iter=500):
    D = np.diag(A)
    R = A - np.diagflat(D)
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x_new, errors
        x = x_new
    return x, errors


def gauss_seidel(A, b, tol=1e-8, max_iter=500):
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ old[i+1:]) / A[i, i]
        error = np.linalg.norm(x - old, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x, errors
    return x, errors


x_j, err_j = jacobi(A, b)
x_gs, err_gs = gauss_seidel(A, b)
x_exact = np.linalg.solve(A, b)
print("Jacobi temperatures:", x_j, "iterations:", len(err_j))
print("Gauss-Seidel temperatures:", x_gs, "iterations:", len(err_gs))
print("Direct solution:", x_exact)

os.makedirs("results", exist_ok=True)
plt.semilogy(err_j, label="Jacobi")
plt.semilogy(err_gs, label="Gauss-Seidel")
plt.xlabel("Iteration")
plt.ylabel("Maximum change")
plt.title("Convergence of iterative temperature solution")
plt.legend()
plt.grid(True)
plt.savefig("results/task_08.png", dpi=150)
plt.show()

# ---------- Changed parameters: tolerance and boundary temperature ----------
for tol in (1e-4, 1e-12):
    print(f"\n[Changed] tol={tol}: Jacobi {len(jacobi(A, b, tol)[1])}, "
          f"Gauss-Seidel {len(gauss_seidel(A, b, tol)[1])} iterations")
b2 = np.array([150., 0., 0., 0., 20.])
print("[Changed] left boundary 150 C:", np.linalg.solve(A, b2))
B = np.array([[1., 2.], [3., 1.]])          # not diagonally dominant
xb, eb = jacobi(B, np.array([3., 4.]), max_iter=50)
print("[Changed] non-dominant matrix after 50 iterations:", xb, "(diverging)")

# ---------- Modified case: metal rod, 7 interior points, ends 80 C and 30 C ----------
n = 7
A7 = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
b7 = np.zeros(n)
b7[0], b7[-1] = 80, 30
print("\n[Modified] rod temperatures:", np.linalg.solve(A7, b7))
print("[Modified] Jacobi", len(jacobi(A7, b7)[1]), "GS", len(gauss_seidel(A7, b7)[1]))
