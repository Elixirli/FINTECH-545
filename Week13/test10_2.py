import numpy as np
import pandas as pd


def risk_parity_weights_with_budgets(Sigma: np.ndarray,
                                     budgets: np.ndarray,
                                     tol: float = 1e-12,
                                     maxiter: int = 10_000) -> np.ndarray:
    """
    Risk Parity under normal assumption with risk budgets.

    RC_i ∝ w_i * (Sigma * w)_i
    Adjusted RC_i = RC_i / budgets[i]
    Risk parity target: all adjusted RC_i equal.

    Multiplicative update:
        w_i(new) = w_i(old) * (target_RC / adjusted_RC_i)
    """
    n = Sigma.shape[0]
    assert Sigma.shape[1] == n, "Sigma must be a square covariance matrix"
    assert len(budgets) == n, "Budget vector must match number of assets"

    # Start with equal weights
    w = np.full(n, 1.0 / n, dtype=float)

    for _ in range(maxiter):
        Sigma_w = Sigma @ w

        # Standard variance-based risk contribution
        RC = w * Sigma_w

        # Adjust for risk budgets
        RC_adj = RC / budgets

        # Risk parity target: mean adjusted risk contribution
        target = RC_adj.mean()

        # Multiplicative update
        w_new = w * (target / RC_adj)
        w_new /= w_new.sum()

        # Convergence check
        if np.max(np.abs(w_new - w)) < tol:
            return w_new

        w = w_new

    return w


def load_covariance(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    Sigma = df.to_numpy(dtype=float)
    return Sigma


def save_output_weights(weights: np.ndarray,
                        path: str,
                        decimals: int = 6) -> None:
    out = pd.DataFrame({"W": weights})
    out = out.round(decimals)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    in_path = "test5_2.csv"
    out_path = "testout10_2.csv"

    Sigma = load_covariance(in_path)

    # Risk budgets: asset 5 = 0.5
    n = Sigma.shape[0]
    budgets = np.ones(n)
    budgets[-1] = 0.5

    w_rp = risk_parity_weights_with_budgets(Sigma, budgets)

    save_output_weights(w_rp, out_path, decimals=6)
