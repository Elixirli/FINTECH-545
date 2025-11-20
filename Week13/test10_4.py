import numpy as np
import pandas as pd
from scipy.optimize import minimize


def max_sharpe_ratio(mu: np.ndarray,
                     Sigma: np.ndarray,
                     rf: float,
                     tol: float = 1e-12) -> np.ndarray:
    """
    Maximize Sharpe Ratio under normal assumption, long-only (w >= 0), sum(w)=1.

    max Sharpe = (w' (mu - rf)) / sqrt(w' Sigma w)
    We minimize negative Sharpe for numerical stability.
    """

    n = len(mu)

    # Objective: negative Sharpe ratio
    def neg_sharpe(w):
        ret = w @ mu
        var = w @ (Sigma @ w)
        if var <= 0:
            return 1e6
        sharpe = (ret - rf) / np.sqrt(var)
        return -sharpe

    # Constraints: sum(w) = 1
    cons = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    })

    # Bounds: 0.1 <= w <= 0.5
    bnds = [(0.1, 0.5) for _ in range(n)]

    # Initial guess: equal weights
    w0 = np.full(n, 1.0 / n)

    sol = minimize(neg_sharpe, w0,
                   bounds=bnds,
                   constraints=cons,
                   method="SLSQP",
                   tol=tol)

    return sol.x


def load_covariance(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.to_numpy(float)


def load_means(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=0)
    mu = df.values.flatten().astype(float)
    return mu


def save_output_weights(w: np.ndarray, path: str):
    out = pd.DataFrame({"W": w})
    out.to_csv(path, index=False)


if __name__ == "__main__":
    cov_path = "test5_2.csv"
    means_path = "test10_3_means.csv"
    out_path = "testout10_4.csv"

    Sigma = load_covariance(cov_path)
    mu = load_means(means_path)
    rf = 0.04

    w_star = max_sharpe_ratio(mu, Sigma, rf)

    save_output_weights(w_star, out_path)