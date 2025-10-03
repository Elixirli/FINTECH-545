import numpy as np
import pandas as pd


def simulate_normal(data: np.ndarray, n_sim: int = 100000, seed: int = 42):
    '''A generic function to directly read data'''
    np.random.seed(seed)
    n = data.shape[0]
    mean = np.zeros(n)
    sims = np.random.multivariate_normal(mean, data, size=n_sim)
    return np.cov(sims, rowvar=False)


def load_csv_matrix(path: str):
    '''Read csv file'''
    df = pd.read_csv(path, header=0)
    df = df.dropna()
    return df.values


def save_csv_matrix(mat: np.ndarray, path: str):
    '''Save output file'''
    n = mat.shape[0]
    colnames = [f"x{i+1}" for i in range(n)]
    df = pd.DataFrame(mat, columns=colnames)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    data = load_csv_matrix("test5_1.csv")
    out = simulate_normal(data, 100000, seed=42)
    save_csv_matrix(out, "testout_5.1.csv")
