import pandas as pd
import numpy as np

from test2_2 import ew_correlation

def ew_variance(X: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Calculate EW variance and then transform variance to standard deviation.
    """
    n, m = X.shape

    weights = np.array([(1 - lambda_) * (lambda_ ** i) for i in range(n)])
    weights = weights[::-1]
    weights /= weights.sum()

    mu = np.average(X, axis=0, weights=weights)

    Z = X - mu
    cov = np.einsum('t,ti,tj->ij', weights, Z, Z)

    std = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    return std


def ew_covariance_varcorr(input_file, output_file, lambda_var=0.97, lambda_corr=0.94):
    """
    Calculate covariance matrix with EW Variance (λ=0.97) and EW Correlation (λ=0.94).
    """
    df = pd.read_csv(input_file).dropna()
    X = df.values
    cols = df.columns

    std97 = ew_variance(X, lambda_var)
    D97 = np.diag(std97)

    corr94 = ew_correlation(input_file, lambda_corr)
    if isinstance(corr94, pd.DataFrame):
        corr94 = corr94.values  

    cov23 = D97 @ corr94 @ D97
    cov23 = 0.5 * (cov23 + cov23.T)  

    cov_df = pd.DataFrame(cov23, index=cols, columns=cols)
    cov_df.to_csv(output_file)


if __name__ == "__main__":
    input_file = "test2.csv"
    output_file = "testout_2.3.csv"
    ew_covariance_varcorr(input_file, output_file, lambda_var=0.97, lambda_corr=0.94)