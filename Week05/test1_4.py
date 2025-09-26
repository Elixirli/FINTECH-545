import pandas as pd


def correlation_pairwise(input_file, output_file):
    df = pd.read_csv(input_file)
    corr_matrix = df.corr()
    corr_matrix.to_csv(output_file)


if __name__ == "__main__":
    input_file = "test1.csv"
    output_file = "testout_1.4.csv"
    correlation_pairwise(input_file, output_file)