import numpy as np
import pandas as pd
from scipy.stats import norm


def var_from_returns(returns: np.ndarray, alpha: float = 0.05):
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)  
    q_alpha = norm.ppf(alpha)        
    var_abs = -(mu + sigma * q_alpha)
    var_diff = -(sigma * q_alpha)
    return var_abs, var_diff, mu, sigma


def load_returns(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.values.flatten()


def save_var(var_abs: float, var_diff: float, path: str):
    df = pd.DataFrame({
        "VaR Absolute": [var_abs],
        "VaR Diff from Mean": [var_diff]
    })
    df.to_csv(path, index=False)


if __name__ == "__main__":
    returns = load_returns("test7_1.csv")
    var_abs, var_diff, mu, sigma = var_from_returns(returns, alpha=0.05)
    save_var(var_abs, var_diff, "testout_8.1.csv")