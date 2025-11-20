import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def expost_attribution(returns: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    assets = returns.columns
    w = weights.flatten()

    # 1) Total Return per asset
    total_ret = returns.sum(axis=0)

    # portfolio total
    total_port = np.dot(w, total_ret)

    # 2) Return Attribution (sum w_i r_i_t)
    ret_attr = (returns * w).sum(axis=0)
    ret_attr_port = ret_attr.sum()

    # 3) Vol Attribution (w_i * std_i)
    stds = returns.std(axis=0, ddof=1)  # sample std
    vol_attr = w * stds
    vol_attr_port = vol_attr.sum()

    # Construct output DataFrame
    out = pd.DataFrame({
        "Value": ["TotalReturn", "Return Attribution", "Vol Attribution"]
    })

    for a in assets:
        out[a] = [
            total_ret[a],
            ret_attr[a],
            vol_attr[assets.get_loc(a)]
        ]

    # Add portfolio column
    out["Portfolio"] = [total_port, ret_attr_port, vol_attr_port]

    return out


def save_output(df: pd.DataFrame, path: str, decimals=6):
    df = df.round(decimals)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    returns = load_csv("test11_1_returns.csv")
    weights_df = load_csv("test11_1_weights.csv")
    weights = weights_df["W"].to_numpy()

    result = expost_attribution(returns, weights)
    save_output(result, "testout_11.1.csv")