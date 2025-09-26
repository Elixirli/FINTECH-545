import pandas as pd


def calculate_arithmetic_returns(input_file, output_file):
    df = pd.read_csv(input_file, index_col=0)
    returns = df.pct_change().dropna()
    returns.to_csv(output_file)


if __name__ == "__main__":
    input_file = "test6.csv"
    output_file = "testout6_1.csv"
    calculate_arithmetic_returns(input_file, output_file)
