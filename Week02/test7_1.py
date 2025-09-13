import pandas as pd
import numpy as np


def normal_distribution(input="test7_1.csv", output="testout7_1.csv"):
    x1 = pd.read_csv(input).iloc[:, 0].values
    mu = np.mean(x1)
    sigma = np.std(x1, ddof=1)
    pd.DataFrame({"mu": [mu], "sigma": [sigma]}).to_csv(output, index=False)

if __name__ == "__main__":
    normal_distribution()