import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=0)


def expost_factor_attribution(F: pd.DataFrame,
                              R: pd.DataFrame,
                              beta_df: pd.DataFrame,
                              weights: np.ndarray) -> pd.DataFrame:
    """
    Ex-post factor attribution following Week 08 lecture:

    INPUTS:
        F: factor return matrix (T × K)
        R: stock return matrix  (T × N)
        beta_df: stock factor loadings (N × (1+K)), with a 'Stock' column
        weights: portfolio weights (N,)

    OUTPUT:
        DataFrame with:
            TotalReturn
            Return Attribution
            Vol Attribution
        Columns = factor names + Alpha + Portfolio

    METHOD:
    ------------------------------------------------------------
    1) Portfolio returns:
           p_t = R_t · w

    2) Portfolio factor exposures:
           B_k = sum_i ( w_i * beta_{i,k} )

    3) Factor-driven portfolio returns:
           p_factor_t = F_t · B

    4) Return Attribution:
           Factor_k = B_k * sum_t(F_{t,k})
           Alpha    = sum_t(p_t) - sum_k(Factor_k)

    5) Volatility Attribution:
        - Portfolio volatility       = std( p_t )
        - Factor volatilities        = std( F_k )
        - Residual volatility        = std( p_t - p_factor_t )
        - Factor contribution        = |B_k| * sigma(F_k)
        - Alpha contribution         = sigma(residual)
        - Rescale so that:
              sum(FactorVol) + AlphaVol = sigma(portfolio)
    ------------------------------------------------------------
    """

    # Convert tables to numeric arrays
    Fv = F.to_numpy(float)                 # (T × K)
    Rv = R.to_numpy(float)                 # (T × N)
    beta_v = beta_df.iloc[:, 1:].to_numpy(float)   # drop the 'Stock' column
    w = weights.astype(float).flatten()    # (N,)

    # === 1) Portfolio returns p_t
    p_t = Rv @ w                           # (T,)

    # === 2) Portfolio factor exposures
    B = w @ beta_v                         # (K,)

    # === 3) Factor-driven returns for the portfolio
    p_factor_t = Fv @ B                    # (T,)

    # === 4) Return Attribution
    F_sum = Fv.sum(axis=0)                 # total return per factor
    FA_factors = B * F_sum                 # contribution of each factor
    total_port_ret = p_t.sum()
    alpha_ret = total_port_ret - FA_factors.sum()

    # === 5) Volatility Attribution
    sigma_p = p_t.std(ddof=1)              # portfolio volatility
    sigma_f = Fv.std(axis=0, ddof=1)       # per-factor volatility
    sigma_res = (p_t - p_factor_t).std(ddof=1)  # residual volatility

    # raw (unscaled) factor vol contributions
    VA_factors = np.abs(B) * sigma_f
    raw_vol_sum = VA_factors.sum() + sigma_res

    # scale contributions to match total portfolio volatility
    if raw_vol_sum > 0:
        scale = sigma_p / raw_vol_sum
    else:
        scale = 1.0

    VA_factors_scaled = VA_factors * scale
    alpha_vol = sigma_res * scale
    portfolio_vol = sigma_p               # final portfolio sigma

    # === Build Output Table ===
    factor_names = F.columns.tolist()
    columns = factor_names + ["Alpha", "Portfolio"]

    out = pd.DataFrame({
        "Value": [
            "TotalReturn",
            "Return Attribution",
            "Vol Attribution"
        ]
    })

    # TotalReturn row
    row1 = list(F_sum) + [alpha_ret, total_port_ret]

    # Return Attribution row
    row2 = list(FA_factors) + [alpha_ret, total_port_ret]

    # Vol Attribution row
    row3 = list(VA_factors_scaled) + [alpha_vol, portfolio_vol]

    out = pd.concat([out, pd.DataFrame([row1, row2, row3], columns=columns)], axis=1)
    return out


def save_output(df: pd.DataFrame, path: str, decimals=6):
    df = df.round(decimals)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    F = load_csv("test11_2_factor_returns.csv")     # factor returns
    R = load_csv("test11_2_stock_returns.csv")      # stock returns
    beta_df = load_csv("test11_2_beta.csv")         # betas
    weights = load_csv("test11_2_weights.csv").iloc[:, 0].to_numpy(float)

    result = expost_factor_attribution(F, R, beta_df, weights)
    save_output(result, "testout_11.2.csv")
