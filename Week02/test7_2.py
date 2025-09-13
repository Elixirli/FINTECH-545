import pandas as pd
from scipy.stats import t


def t_distribution(input="test7_2.csv", output="testout7_2.csv"):
    x = pd.read_csv(input).iloc[:, 0].values
    df, mu, sigma = t.fit(x)  
    pd.DataFrame({"mu": [mu], "sigma": [sigma], "nu": [df]}).to_csv(output, index=False)

if __name__ == "__main__":
    t_distribution()