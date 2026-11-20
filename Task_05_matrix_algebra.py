import numpy as np

# ---------- Brief program ----------
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

angle = np.deg2rad(45)
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
scale = np.array([
    [2.0, 0.0],
    [0.0, 1.5]
])

transformation = rotation @ scale
transformed_points = points @ transformation.T   # row vectors need transpose

print("Original points:\n", points)
print("Transformation matrix:\n", transformation)
print("Transformed points:\n", transformed_points)
print("Determinant:", np.linalg.det(transformation))

# ---------- Changed parameter: rotate 90 degrees ----------
a = np.deg2rad(90)
R90 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
print("\n[Changed] 90 degrees:\n", np.round(points @ (R90 @ scale).T, 4))
print("[Changed] scale factor 0 -> det:", np.linalg.det(rotation @ np.diag([2.0, 0.0])))

# ---------- Modified case: robot-arm waypoints rotated 30 deg, scaled 0.8 ----------
path = np.array([[0, 0], [2, 0], [2, 1], [0, 1.0]])
a = np.deg2rad(30)
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
M = R @ (0.8 * np.eye(2))
print("\n[Modified] new path:\n", path @ M.T)
print("[Modified] determinant (area factor):", np.linalg.det(M))
