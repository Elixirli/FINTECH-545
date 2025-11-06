import numpy as np
import pandas as pd
from scipy.stats import norm


def gbsm_european(S, K, r, q, sigma, T, is_call):
    S     = np.asarray(S, dtype=float)
    K     = np.asarray(K, dtype=float)
    r     = np.asarray(r, dtype=float)
    q     = np.asarray(q, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T     = np.asarray(T, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    n = S.shape[0]
    value = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    gamma = np.full(n, np.nan)
    vega  = np.full(n, np.nan)
    rho   = np.full(n, np.nan)
    theta = np.full(n, np.nan)

    ok = (S > 0) & (K > 0) & (sigma > 0) & (T > 0)
    if not np.any(ok):
        return value, delta, gamma, vega, rho, theta

    b  = r - q
    sig_sqrtT = sigma * np.sqrt(T)

    d1 = np.empty(n); d2 = np.empty(n)
    d1[ok] = (np.log(S[ok] / K[ok]) + (b[ok] + 0.5 * sigma[ok]**2) * T[ok]) / (sig_sqrtT[ok])
    d2[ok] = d1[ok] - sig_sqrtT[ok]

    Nd1 = np.empty(n); Nd2 = np.empty(n)
    Nmd1 = np.empty(n); Nmd2 = np.empty(n)
    nd1 = np.empty(n)

    Nd1[ok]  = norm.cdf(d1[ok])
    Nd2[ok]  = norm.cdf(d2[ok])
    Nmd1[ok] = norm.cdf(-d1[ok])
    Nmd2[ok] = norm.cdf(-d2[ok])
    nd1[ok]  = norm.pdf(d1[ok])

    df_r   = np.empty(n); df_bmr = np.empty(n)
    df_r[ok]   = np.exp(-r[ok] * T[ok])           
    df_bmr[ok] = np.exp((b[ok] - r[ok]) * T[ok]) 

    call_mask = ok & is_call
    put_mask  = ok & (~is_call)

    # Call
    value[call_mask] = S[call_mask] * df_bmr[call_mask] * Nd1[call_mask] \
                       - K[call_mask] * df_r[call_mask] * Nd2[call_mask]
    delta[call_mask] = df_bmr[call_mask] * Nd1[call_mask]
    theta[call_mask] = (
        -(S[call_mask] * df_bmr[call_mask] * nd1[call_mask] * sigma[call_mask]) /
        (2.0 * np.sqrt(T[call_mask]))
        - (b[call_mask] - r[call_mask]) * S[call_mask] * df_bmr[call_mask] * Nd1[call_mask]
        - r[call_mask] * K[call_mask] * df_r[call_mask] * Nd2[call_mask]
    )
    rho[call_mask]   = K[call_mask] * T[call_mask] * df_r[call_mask] * Nd2[call_mask]

    # Put
    value[put_mask] = K[put_mask] * df_r[put_mask] * Nmd2[put_mask] \
                      - S[put_mask] * df_bmr[put_mask] * Nmd1[put_mask]
    delta[put_mask] = df_bmr[put_mask] * (Nd1[put_mask] - 1.0)
    theta[put_mask] = (
        -(S[put_mask] * df_bmr[put_mask] * nd1[put_mask] * sigma[put_mask]) /
        (2.0 * np.sqrt(T[put_mask]))
        + (b[put_mask] - r[put_mask]) * S[put_mask] * df_bmr[put_mask] * Nmd1[put_mask]
        + r[put_mask] * K[put_mask] * df_r[put_mask] * Nmd2[put_mask]
    )
    rho[put_mask]   = -K[put_mask] * T[put_mask] * df_r[put_mask] * Nmd2[put_mask]

    # Common Greeks
    gamma[ok] = (df_bmr[ok] * nd1[ok]) / (S[ok] * sig_sqrtT[ok])
    vega[ok]  = S[ok] * df_bmr[ok] * nd1[ok] * np.sqrt(T[ok])

    return value, delta, gamma, vega, rho, theta


def load_option_table(path: str) -> pd.DataFrame:
    """
    Expect columns:
      ID, Option Type, Underlying, Strike, DaysToMaturity, DayPerYear,
      RiskFreeRate, DividendRate, ImpliedVol
    """
    df = pd.read_csv(path).dropna()
    return df


def save_output(ids, value, delta, gamma, vega, rho, theta, path: str, decimals: int = 6):
    out = pd.DataFrame({
        "ID": ids,
        "Value": value,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Rho": rho,
        "Theta": theta
    })
    
    out["ID"] = out["ID"].astype(int)

    out = out.round(decimals)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    in_path  = "test12_1.csv"
    out_path = "testout12_1.csv"

    df = load_option_table(in_path)

    S     = df["Underlying"].to_numpy()
    K     = df["Strike"].to_numpy()
    r     = df["RiskFreeRate"].to_numpy()
    q     = df["DividendRate"].to_numpy()
    sigma = df["ImpliedVol"].to_numpy()
    T     = (df["DaysToMaturity"] / df["DayPerYear"]).to_numpy()
    is_call = df["Option Type"].astype(str).str.upper().str.startswith("C").to_numpy()

    value, delta, gamma, vega, rho, theta = gbsm_european(S, K, r, q, sigma, T, is_call)
    save_output(df["ID"].to_numpy(), value, delta, gamma, vega, rho, theta, out_path, decimals=6)
