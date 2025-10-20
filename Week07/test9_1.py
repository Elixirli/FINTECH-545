import numpy as np
import pandas as pd
from scipy.stats import norm


def empirical_cdf(x):
    ranks = np.argsort(np.argsort(x))
    return (ranks + 0.5) / len(x)


def simulate_copula_returns(returns, n_sim=100_000, seed=42):
    np.random.seed(seed)
    n, k = returns.shape

    U = np.zeros_like(returns)
    for j in range(k):
        U[:, j] = empirical_cdf(returns[:, j])

    Z = norm.ppf(U)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    corr = np.corrcoef(U, rowvar=False)
    L = np.linalg.cholesky(corr)

    Z_sim = np.random.randn(n_sim, k) @ L.T
    U_sim = norm.cdf(Z_sim)

    sim = np.zeros_like(U_sim)
    for j in range(k):
        sorted_r = np.sort(returns[:, j])
        quantiles = np.linspace(0, 1, len(sorted_r))
        sim[:, j] = np.interp(U_sim[:, j], quantiles, sorted_r)

    return sim


def var_es_pct(x, alpha=0.05):
    q = np.quantile(x, alpha)
    var = -q
    es = -np.mean(x[x <= q])
    return var, es


def load_returns(path):
    df = pd.read_csv(path).select_dtypes(include=[np.number])
    return df.values.astype(float)


def load_portfolio(path):
    df = pd.read_csv(path)
    names = df["Stock"].tolist()
    holdings = df["Holding"].astype(float).values
    prices = df["Starting Price"].astype(float).values
    values = holdings * prices
    weights = values / values.sum()
    return names, values, weights


if __name__ == "__main__":
    returns = load_returns("test9_1_returns.csv")
    names, values, weights = load_portfolio("test9_1_portfolio.csv")

    sim = simulate_copula_returns(returns, n_sim=1_000_000, seed=42)

    var95_pct, es95_pct, var95_amt, es95_amt = [], [], [], []
    for j in range(sim.shape[1]):
        v, e = var_es_pct(sim[:, j], alpha=0.05)
        var95_pct.append(v)
        es95_pct.append(e)
        var95_amt.append(v * values[j])
        es95_amt.append(e * values[j])

    port_ret = sim @ weights
    v_tot, e_tot = var_es_pct(port_ret, alpha=0.05)
    total_value = values.sum()
    var95_pct.append(v_tot)
    es95_pct.append(e_tot)
    var95_amt.append(v_tot * total_value)
    es95_amt.append(e_tot * total_value)
    names.append("Total")

    result = pd.DataFrame({
        "Stock": names,
        "VaR95": var95_amt,
        "ES95": es95_amt,
        "VaR95_Pct": var95_pct,
        "ES95_Pct": es95_pct
    })
    result.to_csv("testout_9.1.csv", index=False)
