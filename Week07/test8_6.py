import numpy as np
import pandas as pd
from scipy.stats import t


def es_t_simulation(returns: np.ndarray, alpha: float = 0.05,
                    n_sim: int = 1_000_000, seed: int = 42):
    r = np.asarray(returns, dtype=float).ravel()
    df_hat, mu_hat, scale_hat = t.fit(r)

    # Monte Carlo sampling
    np.random.seed(seed)
    sims = t.rvs(df_hat, loc=mu_hat, scale=scale_hat, size=n_sim)

    q_emp = np.quantile(sims, alpha)
    
    tail_losses = sims[sims <= q_emp]
    es_emp = np.mean(tail_losses)

    es_abs = -es_emp
    es_diff = -(es_emp - mu_hat)
    return es_abs, es_diff, mu_hat, scale_hat, df_hat, q_emp


def load_returns(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.values.astype(float).ravel()


def save_es(es_abs: float, es_diff: float, path: str):
    pd.DataFrame({
        "ES Absolute": [es_abs],
        "ES Diff from Mean": [es_diff]
    }).to_csv(path, index=False)


if __name__ == "__main__":
    returns = load_returns("test7_2.csv")
    es_abs, es_diff, mu_hat, scale_hat, df_hat, q_emp = es_t_simulation(
        returns, alpha=0.05, n_sim=1_000_000, seed=42)

    save_es(es_abs, es_diff, "testout_8.6.csv")
