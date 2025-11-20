import numpy as np
import pandas as pd


def risk_parity_weights(Sigma: np.ndarray,
                        tol: float = 1e-12,
                        maxiter: int = 10_000) -> np.ndarray:
    """
    Compute long-only risk parity weights under normal assumption.

    Risk measure: portfolio variance sigma_p^2 = w' * Sigma * w.
    Risk contribution of asset i is proportional to:
        RC_i ∝ w_i * (Sigma * w)_i

    Risk parity target: all RC_i are equal.

    We use a simple multiplicative update:
        w_i(new) = w_i(old) * (RC_target / RC_i)
    and renormalize to sum to 1 at each step.
    """
    n = Sigma.shape[0]
    assert Sigma.shape[1] == n, "Sigma must be a square covariance matrix"

    # Start with equal weights
    w = np.full(n, 1.0 / n, dtype=float)

    for _ in range(maxiter):
        # Marginal contribution to variance: Sigma * w
        Sigma_w = Sigma @ w

        # Variance risk contributions: RC_i ∝ w_i * (Sigma * w)_i
        RC = w * Sigma_w

        # Target contribution is the average (risk parity condition)
        target = RC.mean()

        # Multiplicative update: increase weights with too low RC, decrease with too high RC
        w_new = w * (target / RC)

        # Enforce full investment (sum of weights = 1)
        w_new /= w_new.sum()

        # Check convergence
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break

        w = w_new

    return w


def load_covariance(path: str) -> np.ndarray:
    """
    Load covariance matrix from CSV.

    The input test5_2.csv has columns like x1, x2, ..., and
    the rows correspond to the same assets. We just convert
    the numeric table to a NumPy array.
    """
    df = pd.read_csv(path)
    Sigma = df.to_numpy(dtype=float)
    return Sigma


def save_output_weights(weights: np.ndarray,
                        path: str,
                        decimals: int = 6) -> None:
    """
    Save risk parity weights to CSV with a single column 'W'.
    """
    out = pd.DataFrame({"W": weights})
    out = out.round(decimals)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    in_path = "test5_2.csv"
    out_path = "testout10_1.csv"

    Sigma = load_covariance(in_path)

    w_rp = risk_parity_weights(Sigma)

    save_output_weights(w_rp, out_path, decimals=6)
