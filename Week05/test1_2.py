import pandas as pd


def correlation_skip(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.dropna()
    corr_matrix = df.corr()
    corr_matrix.to_csv(output_file)


if __name__ == "__main__":
    input_file = "test1.csv"
    output_file = "testout_1.2.csv"
    correlation_skip(input_file, output_file)