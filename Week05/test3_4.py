import pandas as pd
import numpy as np

def nearest_correlation_higham(R, tol=1e-12, max_iter=5000):
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    Y = R.copy()
    deltaS = np.zeros_like(R)

    for _ in range(max_iter):
        Rk = Y - deltaS
        eigvals, eigvecs = np.linalg.eigh(Rk)
        eigvals_clipped = np.clip(eigvals, 0, None)
        X = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        deltaS = X - Rk

        Y = X.copy()
        np.fill_diagonal(Y, 1.0)

        if np.linalg.norm(Y - X, ord="fro") < tol:
            break

    R_psd = 0.5 * (Y + Y.T)
    np.fill_diagonal(R_psd, 1.0)
    return R_psd


def higham_correlation(input_file, output_file, tol=1e-12, max_iter=5000):
    R_df = pd.read_csv(input_file, index_col=0)
    cols = R_df.columns
    R = R_df.values.astype(float)

    R_psd = nearest_correlation_higham(R, tol=tol, max_iter=max_iter)

    out_df = pd.DataFrame(R_psd, index=cols, columns=cols)
    out_df.to_csv(output_file)


if __name__ == "__main__":
    input_file = "testout_1.4.csv"
    output_file = "testout_3.4.csv"
    higham_correlation(input_file, output_file)