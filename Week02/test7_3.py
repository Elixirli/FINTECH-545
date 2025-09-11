import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from scipy.special import gammaln

def t_regression(input):
    df = pd.read_csv(input)
    y = df.iloc[:, -1].values
    X = np.column_stack([np.ones(len(df)), df.iloc[:, :-1].values])
    n, p = X.shape

    def nll(params):
        alpha = params[0]
        betas = params[1:p]
        log_sigma = params[p]
        log_nu = params[p+1]

        sigma = math.exp(log_sigma)
        nu = math.exp(log_nu) + 2.0  # force ν > 2

        beta = np.concatenate([[alpha], betas])
        resid = y - X @ beta

        z2 = (resid / sigma)**2
        logc = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*math.log(nu*math.pi) - log_sigma
        ll = logc - (nu+1)/2 * np.log1p(z2/nu)
        return -np.sum(ll)

    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_ols
    sigma0 = np.std(resid, ddof=1)
    nu0 = 6.0
    x0 = np.concatenate([beta_ols, [math.log(sigma0), math.log(nu0-2.0)]])

    res = minimize(nll, x0, method="L-BFGS-B")

    alpha = float(res.x[0])
    betas = res.x[1:p].astype(float)
    sigma = float(math.exp(res.x[p]))
    nu = float(math.exp(res.x[p+1]) + 2.0)

    # Build output
    output = {"mu": [0.0], "sigma": [sigma], "nu": [nu], "Alpha": [alpha]}
    for i, b in enumerate(betas, 1):
        output[f"B{i}"] = [b]

    df_out = pd.DataFrame(output)
    print(df_out.to_string(index=False))


t_regression("test7_3.csv")