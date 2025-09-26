import pandas as pd
import numpy as np

def near_psd_correlation(input_file: str, output_file: str, eps: float = 0.0):
    R_df = pd.read_csv(input_file, index_col=0)
    cols = R_df.columns
    R = R_df.values.astype(np.float64)

    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    eigvals, S = np.linalg.eigh(R)
    eigvals_clipped = np.maximum(eigvals, eps)

    s2 = S**2
    denom_t = s2 @ eigvals_clipped
    denom_t = np.where(denom_t > 0, denom_t, 1.0)
    t = 1.0 / np.sqrt(denom_t)
    T = np.diag(t)

    sqrtLambda = np.diag(np.sqrt(eigvals_clipped))
    B = T @ S @ sqrtLambda

    R_star = B @ B.T
    R_star = 0.5 * (R_star + R_star.T)
    np.fill_diagonal(R_star, 1.0)

    out_df = pd.DataFrame(R_star, index=cols, columns=cols)
    out_df.to_csv(output_file)


if __name__ == "__main__":
    input_file = "testout_1.4.csv"
    output_file = "testout_3.2.csv"
    near_psd_correlation(input_file, output_file, eps=0.0)
