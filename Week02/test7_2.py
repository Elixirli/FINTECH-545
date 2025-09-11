import pandas as pd
from scipy.stats import t


def t_distribution(input):
    x = pd.read_csv(input).iloc[:,0].values
    df, mu, sigma = t.fit(x)
    output = {"mu": [mu], "sigma": [sigma], "nu": [df]}
    df_out = pd.DataFrame(output)
    print(df_out.to_string(index=False))


t_distribution("test7_2.csv")