import numpy as np
import pandas as pd


def simulate_normal_psd(data: np.ndarray, n_sim: int = 100_000, seed: int = 42):
    np.random.seed(seed)
    #eigvals, eigvecs = np.linalg.eigh(data)
    #eigvals[eigvals < 0] = 0.0
    #L = eigvecs @ np.diag(np.sqrt(eigvals))
    #Z = np.random.randn(n_sim, data.shape[0])
    #X = Z @ L.T
    #return np.cov(X, rowvar=False)
    sims = np.random.multivariate_normal(np.zeros(data.shape[0]), data, size=n_sim)
    return np.cov(sims, rowvar=False)

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
    data = load_csv_matrix("test5_2.csv")
    out = simulate_normal_psd(data, 100_000, seed=42)
    save_csv_matrix(out, "testout_5.2.csv")
