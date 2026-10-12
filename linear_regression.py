import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(165, 10, (5000, 1))                # (N, d), d bất kỳ
y = -45 + 0.65 * X + rng.normal(0, 3, (5000, 1))

Xb = np.hstack([np.ones((len(X), 1)), X])
w = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
print(w.ravel(), (np.array([[1, 155]]) @ w).item())