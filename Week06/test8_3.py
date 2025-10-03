import numpy as np
import pandas as pd
from scipy.stats import t


def load_returns(path: str):
    df = pd.read_csv(path)
    return df.values.astype(float).ravel()


def var_from_simulation(returns: np.ndarray, alpha: float = 0.05, n_sim: int = 100000, seed: int = 42):
    df_hat, mu_hat, scale_hat = t.fit(returns)

    np.random.seed(seed)
    sims = t.rvs(df_hat, loc=mu_hat, scale=scale_hat, size=n_sim)

    q_emp = np.quantile(sims, alpha)

    var_abs = -q_emp
    var_diff = -(q_emp - mu_hat)
    return var_abs, var_diff


def save_var(var_abs: float, var_diff: float, path: str):
    df = pd.DataFrame({
        "VaR Absolute": [var_abs],
        "VaR Diff from Mean": [var_diff]
    })
    df.to_csv(path, index=False)


if __name__ == "__main__":
    returns = load_returns("test7_2.csv")
    var_abs, var_diff = var_from_simulation(returns, alpha=0.05, n_sim=100000, seed=42)
    save_var(var_abs, var_diff, "testout_8.3.csv")
