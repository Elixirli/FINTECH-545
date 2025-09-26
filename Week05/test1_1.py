import pandas as pd


def covariance_skip(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.dropna()
    cov_matrix = df.cov()
    cov_matrix.to_csv(output_file)


if __name__ == "__main__":
    input_file = "test1.csv"
    output_file = "testout_1.1.csv"
    covariance_skip(input_file, output_file)