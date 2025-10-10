import numpy as np
import pandas as pd
from scipy.stats import t


def es_t(returns: np.ndarray, alpha: float = 0.05):
    r = np.asarray(returns, dtype=float).ravel()
    df_hat, mu_hat, scale_hat = t.fit(r)
    
    t_alpha = t.ppf(alpha, df_hat)
    pdf_alpha = t.pdf(t_alpha, df_hat)
    
    c = (df_hat + t_alpha**2) / ((df_hat - 1) * alpha)
    es_abs  = -mu_hat + scale_hat * pdf_alpha * c
    es_diff =           scale_hat * pdf_alpha * c

    return es_abs, es_diff, mu_hat, scale_hat, df_hat


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
    es_abs, es_diff, mu_hat, scale_hat, df_hat = es_t(returns, alpha=0.05)
    save_es(es_abs, es_diff, "testout_8.5.csv")
