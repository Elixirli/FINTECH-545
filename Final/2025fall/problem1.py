import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


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



returns = pd.read_csv('problem1.csv').to_numpy()
S = 100
r = 0.04
q = 0
sigma = returns.std(ddof=1)* np.sqrt(255)
T = 1/255

# a)
K1 = 99
price_call_99 = gbsm_price(S, K1, r, q, sigma, T, True)
price_put_99 = gbsm_price(S, K1, r, q, sigma, T, False)

K2 = 100
price_call_100 = gbsm_price(S, K2, r, q, sigma, T, True)
price_put_100 = gbsm_price(S, K2, r, q, sigma, T, False)

K3 = 101
price_call_101 = gbsm_price(S, K3, r, q, sigma, T, True)
price_put_101 = gbsm_price(S, K3, r, q, sigma, T, False)

print('Call prices:')
print('Strike = 99: ',  price_call_99)
print('Strike = 100:',  price_call_100)
print('Strike = 101:',  price_call_101)

print('Put prices:')
print('Strike = 99: ',  price_put_99)
print('Strike = 100:',  price_put_100)
print('Strike = 101:',  price_put_101)

# b)
price = []
im_vol = []

for K in range(95, 105):
    price_call = gbsm_price(S, K, r, q, sigma, T, True)
    vol = implied_vol_call(price_call, S, K, r, q, T, 
                           low=1e-4, high=3.0, tol=1e-8, max_iter=200)
    price.append(price_call)
    im_vol.append(vol)

print(price)
print(im_vol)

plt.plot(price, im_vol)
plt.show()


