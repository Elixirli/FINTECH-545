import pandas as pd
import numpy as np


def ew_correlation(input_file, lambda_=0.94, output_file=None):
    df = pd.read_csv(input_file).dropna()
    X = df.values
    n, m = X.shape

    weights = np.array([(1 - lambda_) * (lambda_ ** i) for i in range(n)])
    weights = weights[::-1]  
    weights /= weights.sum() 

    mu = np.average(X, axis=0, weights=weights)

    cov_matrix = np.zeros((m, m))
    for t in range(n):
        diff = (X[t] - mu).reshape(-1, 1)
        cov_matrix += weights[t] * (diff @ diff.T)

    # Transform covariance matrix to correlation matrix
    std_dev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)

    corr_df = pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)
    
    if output_file is not None:
        corr_df.to_csv(output_file)

    return corr_df

if __name__ == "__main__":
    input_file = "test2.csv"
    output_file = "testout_2.2.csv"
    ew_correlation(input_file, 0.94, output_file)