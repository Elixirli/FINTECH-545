import pandas as pd
import numpy as np


def chol_psd(C, tol=1e-12):
    n = C.shape[0]
    L = np.zeros_like(C)

    for i in range(n):
        for j in range(i+1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:  
                val = C[i, i] - s
                if val < tol:
                    val = 0.0
                L[i, j] = np.sqrt(val)
            else:  
                if L[j, j] > tol:
                    L[i, j] = (C[i, j] - s) / L[j, j]
                else:
                    L[i, j] = 0.0
    return L


def chol_psd_from_file(input_file, output_file, tol=1e-12):
    C_df = pd.read_csv(input_file, index_col=0)
    cols = C_df.columns
    C = C_df.values.astype(float)

    L = chol_psd(C, tol=tol)

    out_df = pd.DataFrame(L, index=cols, columns=cols)
    out_df.to_csv(output_file)


if __name__ == "__main__":
    input_file = "testout_3.1.csv"
    output_file = "testout_4.1.csv"
    chol_psd_from_file(input_file, output_file)