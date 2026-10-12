import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def make_data(n_samples=5000, noise_std=3.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(165, 10, size=(n_samples, 1))          # chiều cao (cm)
    true_w0, true_w1 = -45.0, 0.65                        # tham số "thật"
    y = true_w0 + true_w1 * X + rng.normal(0, noise_std, size=(n_samples, 1))
    return X, y, (true_w0, true_w1)


def add_bias(X):
    """Thêm cột 1 vào đầu X -> Xbar có kích thước (N, d+1)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit(X, y, l2=0.0):
    """Nghiệm dạng đóng: w = (XbarᵀXbar + λI)⁻¹ Xbarᵀ y.
    l2 > 0 là Ridge (không phạt hệ số chặn w0)."""
    Xbar = add_bias(X)
    A = Xbar.T @ Xbar
    if l2 > 0:
        reg = l2 * np.eye(A.shape[0])
        reg[0, 0] = 0
        A = A + reg
    b = Xbar.T @ y
    return np.linalg.pinv(A) @ b


def predict(X, w):
    return add_bias(X) @ w


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]

if __name__ == "__main__":
    X, y, (w0_true, w1_true) = make_data(n_samples=5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    w = fit(X_train, y_train)
    print(f"w học được : w0 = {w[0, 0]:.3f}, w1 = {w[1, 0]:.3f}")
    print(f"w thật     : w0 = {w0_true:.3f}, w1 = {w1_true:.3f}")

    y_pred = predict(X_test, w)
    print(f"Test MSE = {mse(y_test, y_pred):.3f} | R² = {r2_score(y_test, y_pred):.3f}")

    for h in (155, 160, 175):
        print(f"Chiều cao {h} cm -> cân nặng dự đoán: "
              f"{predict(np.array([[h]]), w)[0, 0]:.2f} kg")

    plt.scatter(X_test[:300], y_test[:300], s=10, alpha=0.5, label="Dữ liệu test")
    xs = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(xs, predict(xs, w), "r-", label="Đường hồi quy")
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.legend()
    plt.savefig("regression_plot.png", dpi=120)

    rng = np.random.default_rng(1)
    X3 = rng.normal(size=(10000, 3))
    w_true = np.array([[2.0], [1.5], [-3.0], [0.5]])      # w0, w1, w2, w3
    y3 = add_bias(X3) @ w_true + rng.normal(0, 0.5, size=(10000, 1))
    print("Nhiều đặc trưng, w học được:", fit(X3, y3).ravel().round(3))