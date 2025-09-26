import pandas as pd
import numpy as np


def calculate_log_returns(input_file, output_file):
    df = pd.read_csv(input_file, index_col=0)
    log_returns = np.log(df / df.shift(1)).dropna()
    log_returns.to_csv(output_file)


if __name__ == "__main__":
    input_file = "test6.csv"
    output_file = "testout6_2.csv"
    calculate_log_returns(input_file, output_file)