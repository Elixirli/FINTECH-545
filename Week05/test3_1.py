import pandas as pd
import numpy as np

def near_psd_covariance(input_file: str, output_file: str, eps: float = 0.0):
    C_df = pd.read_csv(input_file, index_col=0)
    cols = C_df.columns
    C = C_df.values.astype(np.float64)

    std = np.sqrt(np.clip(np.diag(C), 0.0, np.inf))
    D = np.diag(std)
    denom = np.outer(std, std)
    R = np.divide(C, denom, out=np.zeros_like(C), where=denom > 0)
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

    C_star = D @ R_star @ D
    C_star = 0.5 * (C_star + C_star.T)

    out = pd.DataFrame(C_star, index=cols, columns=cols)
    out.to_csv(output_file)
    

if __name__ == "__main__":
    input_file = "testout_1.3.csv"
    output_file = "testout_3.1.csv"
    near_psd_covariance(input_file, output_file, eps=0.0) 