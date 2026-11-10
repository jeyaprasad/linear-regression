"""Train from-scratch linear regression on California Housing and compare with sklearn."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   # ONLY for the sanity-check comparison

from linreg import LinearRegressionGD, mse, r2_score_manual

SEED = 42
TARGET = "median_house_value"

# ---------------------------------------------------------------- 1. load
df = pd.read_csv("data/housing.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0], "\n")

# one-hot encode the only categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True, dtype=float)

y = df[TARGET].to_numpy(dtype=float)
X_df = df.drop(columns=[TARGET])

# ---------------------------------------------------------------- 2. split
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=SEED
)

# impute missing values with TRAIN medians (no test leakage)
medians = X_train_df.median()
X_train_df = X_train_df.fillna(medians)
X_test_df = X_test_df.fillna(medians)


def standardize(train, test):
    """z-score using TRAIN mean/std only."""
    mu, sigma = train.mean(axis=0), train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (train - mu) / sigma, (test - mu) / sigma


# Scale the target too, so the learning rate behaves and the cost is O(1).
y_mu, y_sigma = y_train.mean(), y_train.std()
y_train_s = (y_train - y_mu) / y_sigma


def run(feature_cols, lr, n_iters, label):
    Xtr = X_train_df[feature_cols].to_numpy(dtype=float)
    Xte = X_test_df[feature_cols].to_numpy(dtype=float)
    Xtr_s, Xte_s = standardize(Xtr, Xte)

    # ---- from scratch
    model = LinearRegressionGD(learning_rate=lr, n_iters=n_iters).fit(Xtr_s, y_train_s)
    pred = model.predict(Xte_s) * y_sigma + y_mu          # back to dollars
    my_mse, my_r2 = mse(y_test, pred), r2_score_manual(y_test, pred)

    # ---- sklearn sanity check (same standardized data, same split)
    sk = LinearRegression().fit(Xtr_s, y_train_s)
    sk_pred = sk.predict(Xte_s) * y_sigma + y_mu
    sk_mse, sk_r2 = mse(y_test, sk_pred), r2_score_manual(y_test, sk_pred)

    max_coef_diff = np.max(np.abs(model.coef_ - sk.coef_))
    print(f"=== {label} ===")
    print(f"lr={lr}, iters={n_iters}, final train cost={model.cost_history[-1]:.6f}")
    print(f"{'':14s}{'MSE':>18s}{'R2':>10s}")
    print(f"{'From scratch':14s}{my_mse:18,.2f}{my_r2:10.6f}")
    print(f"{'sklearn':14s}{sk_mse:18,.2f}{sk_r2:10.6f}")
    print(f"max |coef difference| (standardized space): {max_coef_diff:.2e}")
    print(f"intercept: scratch={model.intercept_:.6f}  sklearn={sk.intercept_:.6f}\n")

    coef_table = pd.DataFrame(
        {"feature": feature_cols, "scratch": model.coef_, "sklearn": sk.coef_}
    )
    return model, coef_table, dict(mse=my_mse, r2=my_r2, sk_mse=sk_mse, sk_r2=sk_r2,
                                   coef_diff=max_coef_diff)


# ---------------------------------------------------------------- 3. simple regression
simple_model, simple_coefs, simple_res = run(
    ["median_income"], lr=0.1, n_iters=500, label="Simple (median_income only)"
)

# ---------------------------------------------------------------- 4. multiple regression
all_features = list(X_df.columns)
multi_model, multi_coefs, multi_res = run(
    all_features, lr=0.3, n_iters=5000, label=f"Multiple ({len(all_features)} features)"
)
print("Coefficients (standardized features):")
print(multi_coefs.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

# ---------------------------------------------------------------- 5. plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, m, title in [
    (axes[0], simple_model, "Simple regression"),
    (axes[1], multi_model, "Multiple regression"),
]:
    ax.plot(m.cost_history)
    ax.set_title(f"{title}: cost vs. iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE cost J(theta)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cost_convergence.png", dpi=150)

# learning-rate comparison to justify the chosen alpha
fig, ax = plt.subplots(figsize=(6, 4))
Xtr_s, _ = standardize(X_train_df.to_numpy(dtype=float), X_test_df.to_numpy(dtype=float))
for lr in [0.001, 0.01, 0.1, 0.5]:
    m = LinearRegressionGD(learning_rate=lr, n_iters=1000).fit(Xtr_s, y_train_s)
    ax.plot(m.cost_history, label=f"alpha={lr}")
ax.set_yscale("log")
ax.set_xlabel("Iteration"); ax.set_ylabel("Cost (log scale)")
ax.set_title("Effect of learning rate"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("learning_rates.png", dpi=150)

# simple-regression fit line
fig, ax = plt.subplots(figsize=(6, 4))
x_raw = X_test_df["median_income"].to_numpy()
mu, sd = X_train_df["median_income"].mean(), X_train_df["median_income"].std()
ax.scatter(x_raw, y_test, s=3, alpha=0.3, label="test data")
xs = np.linspace(x_raw.min(), x_raw.max(), 100)
ax.plot(xs, simple_model.predict((xs - mu) / sd) * y_sigma + y_mu, "r", label="fitted line")
ax.set_xlabel("median_income"); ax.set_ylabel("median_house_value"); ax.legend()
plt.tight_layout()
plt.savefig("simple_fit.png", dpi=150)

pd.to_pickle({"simple": simple_res, "multi": multi_res,
              "coefs": multi_coefs}, "results.pkl")
print("Saved cost_convergence.png, learning_rates.png, simple_fit.png")
