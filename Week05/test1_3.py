import pandas as pd


def covariance_pairwise(input_file, output_file):
    df = pd.read_csv(input_file)
    cov_matrix = df.cov()
    cov_matrix.to_csv(output_file)


if __name__ == "__main__":
    input_file = "test1.csv"
    output_file = "testout_1.3.csv"
    covariance_pairwise(input_file, output_file)