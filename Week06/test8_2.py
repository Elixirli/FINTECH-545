import numpy as np
import pandas as pd
from scipy.stats import t


def var_from_tdist(returns: np.ndarray, alpha: float = 0.05):
    '''MLE method'''
    df_hat, loc_hat, scale_hat = t.fit(returns) 
    q = t.ppf(alpha, df_hat)                      
    var_abs  = -(loc_hat + scale_hat * q)
    var_diff = -(scale_hat * q)
    return var_abs, var_diff, df_hat, loc_hat, scale_hat


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
    returns = load_returns("test7_2.csv")
    var_abs, var_diff, df_hat, loc_hat, scale_hat = var_from_tdist(returns, alpha=0.05)
    save_var(var_abs, var_diff, "testout_8.2.csv")