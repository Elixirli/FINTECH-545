import numpy as np
import pandas as pd

def pca_simulation(S: np.ndarray, explained: float = 0.99, n_sim: int = 100_000, seed: int = 42):
    np.random.seed(seed)
    
    eigvals, eigvecs = np.linalg.eigh(S)
    
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_var = np.sum(eigvals)
    var_ratio = np.cumsum(eigvals) / total_var
    k = np.searchsorted(var_ratio, explained) + 1  

    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:, :k]
    L = eigvecs_k @ np.diag(np.sqrt(eigvals_k))

    Z = np.random.randn(n_sim, k)
    X = Z @ L.T
    return np.cov(X, rowvar=False), k, var_ratio[k-1]


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
    S_in = load_csv_matrix("test5_2.csv")
    S_sim, k, explained_ratio = pca_simulation(S_in, explained=0.99, n_sim=100_000, seed=42)
    save_csv_matrix(S_sim, "testout_5.5.csv")
