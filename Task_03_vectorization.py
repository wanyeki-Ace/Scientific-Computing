import numpy as np
import timeit

# ---------- Brief program: loop versus vectorized ----------
rng = np.random.default_rng(42)
latency_ms = rng.uniform(5, 250, 1_000_000)


def loop_version():
    squared = []
    for value in latency_ms:
        seconds = value / 1000.0
        squared.append(seconds ** 2)
    return squared


def vectorized_version():
    seconds = latency_ms / 1000.0
    return seconds ** 2


loop_time = timeit.timeit(loop_version, number=1)
vector_time = timeit.timeit(vectorized_version, number=3) / 3
result = vectorized_version()

print("First five squared latency values:", result[:5])
print(f"Loop time: {loop_time:.4f} s")
print(f"Vectorized time: {vector_time:.4f} s")
print(f"Approximate speed-up: {loop_time/vector_time:.1f}x")

# ---------- Modified case: gamma correction of a 1000 x 1000 image ----------
# Assumptions: pixel values 0-255; gamma 0.5 brightens dark areas.
rng = np.random.default_rng(42)
img = rng.integers(0, 256, (1000, 1000)).astype(float)
bright = 255 * (img / 255) ** 0.5
print("\n[Modified] Mean pixel before/after:", img.mean(), bright.mean())
