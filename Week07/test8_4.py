import numpy as np
import pandas as pd
from scipy.stats import norm


def es_normal(returns: np.ndarray, alpha: float = 0.05):
    r = np.asarray(returns, dtype=float).ravel()
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    z = norm.ppf(alpha)
    phi = norm.pdf(z)

    es_abs  = -mu + sigma * (phi / alpha)
    es_diff = sigma * (phi / alpha)
    return es_abs, es_diff, mu, sigma


def load_returns(path: str) -> np.ndarray:
    df = pd.read_csv(path)          
    return df.values.astype(float).ravel()


def save_es(es_abs: float, es_diff: float, path: str):
    pd.DataFrame({
        "ES Absolute": [es_abs],
        "ES Diff from Mean": [es_diff]
    }).to_csv(path, index=False)


if __name__ == "__main__":
    rets = load_returns("test7_1.csv")   
    es_abs, es_diff, mu, sigma = es_normal(rets, alpha=0.05)
    save_es(es_abs, es_diff, "testout_8.4.csv")
