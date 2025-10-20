import numpy as np
import pandas as pd

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


def higham_covariance(input_file, tol=1e-12, max_iter=5000):
    C_df = pd.read_csv(input_file)
    cols = C_df.columns
    C = C_df.values.astype(float)

    std = np.sqrt(np.clip(np.diag(C), 0.0, np.inf))
    D = np.diag(std)
    denom = np.outer(std, std)
    R = np.divide(C, denom, out=np.zeros_like(C), where=denom > 0)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    R_psd = nearest_correlation_higham(R, tol=tol, max_iter=max_iter)

    C_psd = D @ R_psd @ D
    C_psd = 0.5 * (C_psd + C_psd.T)

    return C_psd


def simulate_normal_higham(S: np.ndarray, n_sim: int = 100_000, seed: int = 42):
    np.random.seed(seed)
    sims = np.random.multivariate_normal(np.zeros(S.shape[0]), S, size=n_sim)
    return np.cov(sims, rowvar=False)


def save_csv_matrix(mat: np.ndarray, path: str):
    n = mat.shape[0]
    colnames = [f"x{i+1}" for i in range(n)]
    df = pd.DataFrame(mat, columns=colnames)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    C = higham_covariance("test5_3.csv")
    out = simulate_normal_higham(C, 100_000, seed=42)
    save_csv_matrix(out, "testout_5.4_correction.csv")   