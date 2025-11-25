import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math


def ew_covariance(input_file, lambda_=0.97):
    df = pd.read_csv(input_file, index_col=0).dropna()
    X = df.values
    n, m = X.shape

    weights = np.array([(1 - lambda_) * (lambda_ ** i) for i in range(n)])
    weights = weights[::-1]  
    weights /= weights.sum() 

    mu = np.average(X, axis=0, weights=weights)

    cov_matrix = np.zeros((m, m))
    for t in range(n):
        diff = (X[t] - mu).reshape(-1, 1)
        cov_matrix += weights[t] * (diff @ diff.T)

    return cov_matrix


def max_sharpe_ratio(mu: np.ndarray,
                     Sigma: np.ndarray,
                     rf: float,
                     tol: float = 1e-12) -> np.ndarray:
    """
    Maximize Sharpe Ratio under normal assumption, long-only (w >= 0), sum(w)=1.

    max Sharpe = (w' (mu - rf)) / sqrt(w' Sigma w)
    We minimize negative Sharpe for numerical stability.
    """

    n = len(mu)

    # Objective: negative Sharpe ratio
    def neg_sharpe(w):
        ret = w @ mu
        var = w @ (Sigma @ w)
        if var <= 0:
            return 1e6
        sharpe = (ret - rf) / np.sqrt(var)
        return -sharpe

    # Constraints: sum(w) = 1
    cons = ({
        "type": "eq", # equality constraint
        "fun": lambda w: np.sum(w) - 1.0
    })

    # Bounds: w >= 0
    bnds = [(0.0, 1.0) for _ in range(n)]

    # Initial guess: equal weights
    w0 = np.full(n, 1.0 / n)

    sol = minimize(neg_sharpe, w0,
                   bounds=bnds,
                   constraints=cons,
                   method="SLSQP",
                   tol=tol)

    return sol.x


def risk_parity_weights(Sigma: np.ndarray,
                        tol: float = 1e-12,
                        maxiter: int = 10_000) -> np.ndarray:
    n = Sigma.shape[0]
    assert Sigma.shape[1] == n, "Sigma must be a square covariance matrix"

    # Start with equal weights
    w = np.full(n, 1.0 / n, dtype=float)

    for _ in range(maxiter):
        # Marginal contribution to variance: Sigma * w
        Sigma_w = Sigma @ w

        # Variance risk contributions: RC_i ∝ w_i * (Sigma * w)_i
        RC = w * Sigma_w

        # Target contribution is the average (risk parity condition)
        target = RC.mean()

        # Multiplicative update: increase weights with too low RC, decrease with too high RC
        w_new = w * (target / RC)

        # Enforce full investment (sum of weights = 1)
        w_new /= w_new.sum()

        # Check convergence
        if np.max(np.abs(w_new - w)) < tol: # if the weight change is smaller than the tolerance,we can stop
            w = w_new
            break

        w = w_new

    return w


def expost_attribution(returns: pd.DataFrame,
                       weights: np.ndarray) -> pd.DataFrame:
    """
    Ex-post return & risk attribution in one table.

    returns : T x N DataFrame of simple returns (rows = time, cols = assets)
    weights : length-N array of starting weights (sum to 1)

    Output:
        Value, asset1, asset2, ..., Portfolio
        TotalReturn         (asset total returns, portfolio total return)
        Return Attribution  (w_i * TotalReturn_i, sum = portfolio total return)
        Vol Attribution     (volatility RC_i, sum = portfolio sigma)
    """
    assets = list(returns.columns)
    w = np.asarray(weights, dtype=float).flatten()

    # ---------- 1) Total Return per asset (buy-and-hold) ----------
    # TR_i = Π_t (1 + r_{t,i}) - 1
    gross = (1.0 + returns).prod(axis=0)
    total_ret = gross - 1.0                 # Series, index = assets

    # portfolio total return
    total_port = float(np.dot(w, total_ret.to_numpy()))

    # ---------- 2) Return Attribution ----------
    ret_attr = w * total_ret.to_numpy()
    ret_attr_port = ret_attr.sum()
    ret_pct = ret_attr/ret_attr_port

    # ---------- 3) Vol Attribution (covariance-based RC) ----------
    Sigma = returns.cov().to_numpy()        # N x N
    Sigma_w = Sigma @ w
    sigma_p = math.sqrt(float(w @ Sigma_w))

    if sigma_p > 0:
        vol_attr = w * Sigma_w / sigma_p    # volatility units
    else:
        vol_attr = np.zeros_like(w)
    vol_attr_port = vol_attr.sum()          # ≈ sigma_p
    vol_pct = vol_attr/vol_attr_port

    out = pd.DataFrame({"Value": ["TotalReturn",
                                  "Return Attribution",
                                  "Return Contribution %",
                                  "Risk Attribution",
                                  "Risk Contribution %"]})

    for i, a in enumerate(assets):
        out[a] = [total_ret[a],           # row 1
                  ret_attr[i],           # row 2
                  ret_pct[i],
                  vol_attr[i],           # row 3
                  vol_pct[i]]

    out["Portfolio"] = [total_port,
                        ret_attr_port,
                        1,
                        vol_attr_port,
                        1]

    return out


# a)
insample = 'problem3_insample.csv'
cov = ew_covariance(insample, 0.97)

df = pd.read_csv(insample, index_col=0).astype(float)
sigma_m = np.array(cov)
sigma_ann = sigma_m*12
mu_m = df.mean(axis=0).to_numpy()
mu_ann = (1 + mu_m)**12 -1

w_max_sr = max_sharpe_ratio(mu_ann, sigma_ann, 0.04, 1e-12)
w_rp = risk_parity_weights(sigma_ann, 1e-12, 10000)

print('Weights for Maximum Sharpe Portfolio:', w_max_sr)
print('Weights for Risk Parity Portfolio:', w_rp)

# b)
returns = pd.read_csv('problem3_outsample.csv', index_col=0)
returns_ann = (1 + returns)**12 - 1
sr_out = expost_attribution(returns_ann, w_max_sr)
rp_out = expost_attribution(returns_ann, w_rp)
print(sr_out)
print(rp_out)

# ex-ante
ret = pd.read_csv('problem3_insample.csv', index_col=0)
ret_ann = (1 + ret)**12 - 1
ante_sr_out = expost_attribution(ret_ann, w_max_sr)
ante_rp_out = expost_attribution(ret_ann, w_rp)
print(ante_sr_out)
print(ante_rp_out)