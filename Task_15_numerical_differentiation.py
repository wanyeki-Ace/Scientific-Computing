import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- Brief program ----------
time = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
position = np.array([0.0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0])

speed = np.zeros_like(position)
speed[0] = (position[1] - position[0]) / (time[1] - time[0])          # forward
for i in range(1, len(time) - 1):                                      # central
    speed[i] = (position[i + 1] - position[i - 1]) / (time[i + 1] - time[i - 1])
speed[-1] = (position[-1] - position[-2]) / (time[-1] - time[-2])      # backward

for t, x, v in zip(time, position, speed):
    print(f"t={t:.0f}s, position={x:.1f}m, estimated speed={v:.2f}m/s")

os.makedirs("results", exist_ok=True)
plt.plot(time, speed, marker="o")
plt.xlabel("Time (s)")
plt.ylabel("Estimated speed (m/s)")
plt.title("Speed estimated from GPS position data")
plt.grid(True)
plt.savefig("results/task_15.png", dpi=150)
plt.show()

# ---------- Changed parameters ----------
glitch = position.copy()
glitch[3] = 17.5
print("\n[Changed] speeds with glitch at t=3:", np.gradient(glitch, time))

print("\n[Changed] step-size test on sin(x) at x=1")
for h in (1, 0.1, 0.01, 1e-5, 1e-8, 1e-12):
    fwd = (np.sin(1 + h) - np.sin(1)) / h
    cen = (np.sin(1 + h) - np.sin(1 - h)) / (2 * h)
    print(f"h={h:g}: forward error {abs(fwd - np.cos(1)):.2e}, "
          f"central error {abs(cen - np.cos(1)):.2e}")

# ---------- Modified case: drone climb rate ----------
t2 = np.array([0, 10, 20, 30, 40], dtype=float)
alt = np.array([0, 120, 260, 430, 640], dtype=float)
print("\n[Modified] climb rate (m/s):", np.gradient(alt, t2))
