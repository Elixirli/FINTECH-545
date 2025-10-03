import numpy as np
import pandas as pd


def near_psd(S: np.ndarray, epsilon: float = 0):
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals[eigvals < epsilon] = epsilon
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def simulate_normal_near_psd(S: np.ndarray, n_sim=100_000, seed=42):
    '''Normal Simulation with near_psd fix'''
    np.random.seed(seed)
    S_psd = near_psd(S)
    eigvals, eigvecs = np.linalg.eigh(S_psd)
    L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
    Z = np.random.randn(n_sim, S.shape[0])
    X = Z @ L.T
    return np.cov(X, rowvar=False)


def load_csv_matrix(path: str):
    df = pd.read_csv(path, header=0)
    df = df.dropna()
    return df.values.astype(float)


def save_csv_matrix(mat: np.ndarray, path: str):
    n = mat.shape[0]
    colnames = [f"x{i+1}" for i in range(n)]
    df = pd.DataFrame(mat, columns=colnames)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    data = load_csv_matrix("test5_3.csv")
    out = simulate_normal_near_psd(data, 100_000, seed=42)
    save_csv_matrix(out, "testout_5.3.csv")
