import numpy as np
import pandas as pd

def near_psd_covariance(input_file: str, eps: float = 0.0):
    '''Use near_psd_covariance to correct the covariance matrix'''
    C_df = pd.read_csv(input_file)
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

    return C_star


def simulate_normal_near_psd(C: np.ndarray, n_sim=100_000, seed=42):
    sims = np.random.multivariate_normal(np.zeros(C.shape[0]), C, size=n_sim)
    return np.cov(sims, rowvar=False)


def save_csv_matrix(mat: np.ndarray, path: str):
    n = mat.shape[0]
    colnames = [f"x{i+1}" for i in range(n)]
    df = pd.DataFrame(mat, columns=colnames)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    C = near_psd_covariance("test5_3.csv")
    out = simulate_normal_near_psd(C, 100_000, seed=42)
    save_csv_matrix(out, "testout_5.3_correction.csv")