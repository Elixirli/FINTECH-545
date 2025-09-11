import pandas as pd
import numpy as np
from scipy.stats import norm


def normal_distribution(input):
    x1 = pd.read_csv(input).iloc[:,0].values
    mu = np.mean(x1)
    sigma = np.std(x1, ddof=1)
    output = {"mu": [mu], "sigma": [sigma]}
    df_out = pd.DataFrame(output)
    print(df_out.to_string(index=False))


normal_distribution("test7_1.csv")