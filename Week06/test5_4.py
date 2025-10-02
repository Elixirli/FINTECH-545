import numpy as np
import pandas as pd


def near_psd_higham(corr, tol=1e-8, max_iter=100):
    n = corr.shape[0]
    Y = corr.copy()
    delta_S = np.zeros_like(corr)
    gamma = np.inf

    for _ in range(max_iter):
        R = Y - delta_S
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals[eigvals < 0] = 0
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        delta_S = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)  
        gamma_new = np.linalg.norm(Y - corr, "fro")
        if abs(gamma_new - gamma) < tol:
            break
        gamma = gamma_new
    return Y


def cov_to_corr(cov: np.ndarray):
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr, d


def corr_to_cov(corr: np.ndarray, d: np.ndarray):
    return corr * np.outer(d, d)


def simulate_normal_higham(S: np.ndarray, n_sim: int = 100_000, seed: int = 42):
    np.random.seed(seed)
    corr, d = cov_to_corr(S)
    corr_psd = near_psd_higham(corr)
    S_psd = corr_to_cov(corr_psd, d)
    eigvals, eigvecs = np.linalg.eigh(S_psd)
    L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
    Z = np.random.randn(n_sim, S.shape[0])
    X = Z @ L.T
    return np.cov(X, rowvar=False)


def load_csv_matrix(path: str):
    df = pd.read_csv(path, header=0)
    df = df.dropna(how="all")
    return df.values.astype(float)


def save_csv_matrix(mat: np.ndarray, path: str):
    n = mat.shape[0]
    colnames = [f"x{i+1}" for i in range(n)]
    df = pd.DataFrame(mat, columns=colnames)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    input = load_csv_matrix("test5_3.csv")
    output = simulate_normal_higham(input, 100_000, seed=42)
    save_csv_matrix(output, "testout_5.4.csv")
