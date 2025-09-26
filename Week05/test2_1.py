import pandas as pd
import numpy as np


def ew_covariance(input_file, output_file, lambda_=0.97):
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

    cov_df = pd.DataFrame(cov_matrix, index=df.columns, columns=df.columns)
    cov_df.to_csv(output_file)


if __name__ == "__main__":
    input_file = "test2.csv"
    output_file = "testout_2.1.csv"
    ew_covariance(input_file, output_file, lambda_=0.97)