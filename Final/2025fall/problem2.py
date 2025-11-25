import numpy as np
import pandas as pd
from scipy import stats
import math
from scipy.stats import norm
from scipy.stats import t


def fit_normal(r):
    mu = r.mean()
    sigma = r.std(ddof=0)  # MLE
    loglik = stats.norm.logpdf(r, loc=mu, scale=sigma).sum()
    return mu, sigma, loglik


def fit_student_t(r):
    # df, loc, scale
    df_hat, loc_hat, scale_hat = stats.t.fit(r)
    loglik = stats.t.logpdf(r, df_hat, loc=loc_hat, scale=scale_hat).sum()
    return df_hat, loc_hat, scale_hat, loglik


def aicc(loglik, k, n):
    """Corrected AIC."""
    return 2 * k - 2 * loglik + (2 * k**2 + 2 * k) / (n - k - 1)


def gbsm_price(S, K, r, q, sigma, T, is_call=True):
    """
    European option price under GBSM with continuous dividend yield q.

    S : spot
    K : strike
    r : risk-free rate (annual, cont. comp.)
    q : dividend yield (annual, cont. comp.)
    sigma : vol
    T : time to maturity (years)
    is_call : True for call, False for put
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if is_call:
        price = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    else:
        price = K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)
    return price


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def implied_vol_call(mkt_price, S, K, r, q, T,
                     low=1e-4, high=3.0, tol=1e-8, max_iter=200):
    """
    Implied vol for a call under GBSM with dividend yield q.
    Solve: C_BS(S,K,r,q,sigma,T) = mkt_price
    """

    def f(sig):
        return gbsm_price(S, K, r, q, sig, T, is_call=True) - mkt_price

    f_low, f_high = f(low), f(high)
    # expand if needed
    k = 0
    while f_low * f_high > 0 and k < 50:
        high *= 2.0
        f_high = f(high)
        k += 1

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_mid * f_low > 0:
            low, f_low = mid, f_mid
        else:
            high, f_high = mid, f_mid
    return 0.5 * (low + high)


def implied_vol_put(mkt_price, S, K, r, q, T,
                     low=1e-4, high=3.0, tol=1e-8, max_iter=200):
    """
    Implied vol for a put under GBSM with dividend yield q.
    Solve: C_BS(S,K,r,q,sigma,T) = mkt_price
    """

    def f(sig):
        return gbsm_price(S, K, r, q, sig, T, is_call=False) - mkt_price

    f_low, f_high = f(low), f(high)
    # expand if needed
    k = 0
    while f_low * f_high > 0 and k < 50:
        high *= 2.0
        f_high = f(high)
        k += 1

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_mid * f_low > 0:
            low, f_low = mid, f_mid
        else:
            high, f_high = mid, f_mid
    return 0.5 * (low + high)



def var_from_simulation(returns: np.ndarray, alpha: float = 0.05, n_sim: int = 100000, seed: int = 42):
    df_hat, mu_hat, scale_hat = t.fit(returns)

    np.random.seed(seed)
    sims = t.rvs(df_hat, loc=mu_hat, scale=scale_hat, size=n_sim)

    q_emp = np.quantile(sims, alpha)

    var_abs = -q_emp
    var_diff = -(q_emp - mu_hat)
    return var_abs, var_diff


def es_t_simulation(returns: np.ndarray, alpha: float = 0.05,
                    n_sim: int = 1_000_000, seed: int = 42):
    r = np.asarray(returns, dtype=float).ravel()
    df_hat, mu_hat, scale_hat = t.fit(r)

    # Monte Carlo sampling
    np.random.seed(seed)
    sims = t.rvs(df_hat, loc=mu_hat, scale=scale_hat, size=n_sim)

    q_emp = np.quantile(sims, alpha)
    
    tail_losses = sims[sims <= q_emp]
    es_emp = np.mean(tail_losses)

    es_abs = -es_emp
    es_diff = -(es_emp - mu_hat)
    return es_abs, es_diff


# a)
df = pd.read_csv('problem2.csv', index_col=0)
returns = df['SPY'].pct_change().dropna()
n = len(returns)

mu, sigma, ll_N = fit_normal(returns)
df_t, loc_t, scale_t, ll_t = fit_student_t(returns)
aicc_N = aicc(ll_N, k=2, n=n)
aicc_t = aicc(ll_t, k=3, n=n)

print("\nNormal fit:")
print(f"  mu    = {mu:.6f}")
print(f"  sigma = {sigma:.6f}")
print(f"  loglik= {ll_N:.3f}")
print(f"  AICc  = {aicc_N:.3f}")

print("\nStudent-t fit:")
print(f"  df    = {df_t:.3f}")
print(f"  loc   = {loc_t:.6f}")
print(f"  scale = {scale_t:.6f}")
print(f"  loglik= {ll_t:.3f}")
print(f"  AICc  = {aicc_t:.3f}")

better = "Student-t" if aicc_t < aicc_N else "Normal"
print(f"\nBetter model by AICc: {better}")

# b)
call_price = 7.05
S = 659.030029296875
K = 665
r = 0.04
q = 0.0109
T = 10/255
vol_call = implied_vol_call(call_price, S, K, r, q, T,
                     low=1e-4, high=3.0, tol=1e-8, max_iter=200)
print('Implied Volatility for Call Option:', vol_call)

put_price = 7.69
S = 659.030029296875
K = 655
r = 0.04
q = 0.0109
T = 10/255
vol_put = implied_vol_put(put_price, S, K, r, q, T,
                     low=1e-4, high=3.0, tol=1e-8, max_iter=200)
print('Implied Volatility for Put Option:', vol_put)

# c)
T1 = 1/255
T2 = 2/255
T3 = 3/255
T4 = 4/255
T5 = 5/255
ret1 = (K / K*math.exp(-r * T5)) - 1
ret2 = (K / K*math.exp(-r * T4)) - 1
ret3 = (K / K*math.exp(-r * T3)) - 1
ret4 = (K / K*math.exp(-r * T2)) - 1
ret5 = (K / K*math.exp(-r * T1)) - 1
ret = np.array([ret1, ret2, ret3, ret4, ret5])
print(ret)
var_abs, var_diff = var_from_simulation(ret, 0.05)
print(var_abs)
print(var_diff)
es_abs, es_diff = es_t_simulation(ret, 0.05)
print(es_abs)
print(es_diff)

# d)
def simulate_stock_terminal(S0: float,
                            mu: float,
                            sigma: float,
                            T: float,
                            n_paths: int,
                            seed: int = 0) -> np.ndarray:
    """
    Simulate terminal stock prices under GBM with *real-world* drift mu.

    S_T = S0 * exp( (mu - 0.5 sigma^2) T + sigma sqrt(T) Z ),  Z ~ N(0,1).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    drift = (mu - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T) * Z
    S_T = S0 * np.exp(drift + diffusion)
    return S_T


def compute_returns_stock_put_call(S_T: np.ndarray,
                              S0: float,
                              K: float,
                              call_0,
                              put_0: float) -> tuple[np.ndarray, np.ndarray]:
    # Stock returns
    R_stock = (S_T - S0) / S0

    # Put payoff at maturity (European): max(K - S_T, 0)
    put_T = np.maximum(K - S_T, 0.0)
    R_put = (put_T - put_0) / put_0

    call_T = np.maximum(S_T-K, 0.0)
    R_call = (call_T - call_0) / call_0

    return R_stock, R_put, R_call


def portfolio_return(R_stock: np.ndarray,
                     R_put: np.ndarray,
                     R_call,
                     w_stock: float,
                     w_call,
                     w_put: float) -> np.ndarray:
    """
    Portfolio one-year return for given weights.
    """
    return w_stock * R_stock + w_put * R_put + w_call * R_call


def ratio_from_sims(R_p: np.ndarray, rf: float, alpha: float = 0.05) -> float:
    mean_r = R_p.mean()
    # Lower alpha-quantile of returns (this will be a negative number if there are losses)
    q_alpha = np.quantile(R_p, alpha)
    tail = R_p[R_p <= q_alpha]
    if len(tail) == 0:
        return -1e9
    es_loss = -tail.mean()  # positive number
    if es_loss <= 0:
        return -1e9
    return (mean_r - rf) / es_loss


S0 = S        # initial stock price
K1 = 665
K2 = 655         
r = 0.04*5/255                           
q = 0.0           
T = 5/255           

# Price the European put at t=0 under risk-neutral BS
call_0 = gbsm_price(S0, K1, r, q, sigma, T, True)
put_0 = gbsm_price(S0, K2, r, q, sigma, T, False)

S_T = simulate_stock_terminal(S0, mu, sigma, T, 100000, 0)
R_stock, R_put, R_call = compute_returns_stock_put_call(S_T, S0, K, call_0, put_0)

w_put_grid = np.linspace(-2.0, 2.0, 301) 
best_ratio = -1e9
best_w_put = None
best_w_stock = None

for w_put in w_put_grid:
    w_stock = 1.0 - w_put
    # enforce both weights >= -1
    if w_stock < -1.0 or w_put < -1.0:
        continue

    R_p = portfolio_return(R_stock, R_put, R_call, w_stock, w_put)
    ratio = ratio_from_sims(R_p, rf=r)  # rf is annual, same horizon
    if ratio > best_ratio:
        best_ratio = ratio
        best_w_put = w_put
        best_w_stock = w_stock