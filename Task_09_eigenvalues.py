import numpy as np

# ---------- Brief program ----------
A = np.array([
    [0., 1., 1., 0.],
    [1., 0., 1., 1.],
    [1., 1., 0., 1.],
    [0., 1., 1., 0.]
])

eigenvalues, eigenvectors = np.linalg.eig(A)

index = np.argmax(np.real(eigenvalues))
dominant_value = np.real(eigenvalues[index])
dominant_vector = np.abs(np.real(eigenvectors[:, index]))
centrality = dominant_vector / dominant_vector.sum()

nodes = ["A", "B", "C", "D"]
print("Dominant eigenvalue:", dominant_value)
print("Eigenvector-based importance:")
for node, score in zip(nodes, centrality):
    print(f"{node}: {score:.4f}")

v = np.real(eigenvectors[:, index])
print("Verification error:", np.linalg.norm(A @ v - dominant_value * v))

# ---------- Changed parameter: add the missing edge A-D ----------
A2 = A.copy()
A2[0, 3] = A2[3, 0] = 1
w, vec = np.linalg.eig(A2)
i = np.argmax(w.real)
d = np.abs(vec[:, i].real)
print("\n[Changed] with edge A-D: eigenvalue", w.real[i], "scores", d / d.sum())

# ---------- Modified case 1: star network (hub + 4 leaves) ----------
S = np.zeros((5, 5))
S[0, 1:] = 1
S[1:, 0] = 1
w, vec = np.linalg.eigh(S)
score = np.abs(vec[:, -1])
print("\n[Modified] star eigenvalue:", w[-1], "scores:", score / score.sum())

# ---------- Modified case 2: principal component analysis idea ----------
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2)) @ np.array([[3, 1], [0, 1.0]])
w, vec = np.linalg.eigh(np.cov(X.T))
print("[Modified] variance ratio of components:", w[::-1] / w.sum())
