import numpy as np
import pandas as pd


def bt_american(is_call: bool, S: float, K: float, T: float,
                r: float, b: float, sigma: float, steps: int = 500) -> float:
    """
    American option via CRR binomial tree, using carry b in the RN drift:
        p = (exp(b*dt) - d) / (u - d), discount = exp(-r*dt)
    Matches the Julia bt_american(call, S, K, T, r, b, sigma, steps).
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or steps < 2:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    a = np.exp(b * dt)                  
    p = (a - d) / (u - d)
    if p < 0.0 or p > 1.0:
        return float("nan")
    disc = np.exp(-r * dt)
 
    j = np.arange(steps + 1)
    ST = S * (u ** j) * (d ** (steps - j))
    V = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)

    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[1:i+2] + (1.0 - p) * V[0:i+1])
        S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
        exercise = np.maximum(S_i - K, 0.0) if is_call else np.maximum(K - S_i, 0.0)
        V = np.maximum(V, exercise)

    return float(V[0])


def finite_difference_gradient(f, x, rel_step=1e-4, abs_floor=1e-6):
    """
    Central finite-difference gradient of scalar f at vector x (NumPy array-like).
    Step per-dimension: h_i = max(|x_i| * rel_step, abs_floor).
    """
    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x)
    for i in range(len(x)):
        hi = max(abs(x[i]) * rel_step, abs_floor)
        xp = x.copy(); xm = x.copy()
        xp[i] += hi; xm[i] -= hi
        fp = f(xp);  fm = f(xm)
        grad[i] = (fp - fm) / (2.0 * hi)
    return grad


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": "Int64"})
    df = df.dropna(subset=["ID"])
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
    out["ID"] = pd.Series(out["ID"], dtype="Int64")
    out = out.round(decimals)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    in_path  = "test12_1.csv"
    out_path = "testout12_2.csv"

    options = load_table(in_path)

    out_vals = []
    for _, o in options.iterrows():
        is_call = str(o["Option Type"]).strip().lower().startswith("c")
        S = float(o["Underlying"])
        K = float(o["Strike"])
        T = float(o["DaysToMaturity"]) / float(o["DayPerYear"])
        r = float(o["RiskFreeRate"])
        q = float(o["DividendRate"])
        b = r - q
        sigma = float(o["ImpliedVol"])
        price = bt_american(is_call, S, K, T, r, b, sigma, steps=500)
        out_vals.append(price)

    def fcall(parms):
        S, K, T, r, b, sig = map(float, parms)
        return bt_american(True, S, K, T, r, b, sig, steps=500)

    def fput(parms):
        S, K, T, r, b, sig = map(float, parms)
        return bt_american(False, S, K, T, r, b, sig, steps=500)

    deltas, gammas, vegas, rhos, thetas = [], [], [], [], []

    for _, o in options.iterrows():
        is_call = str(o["Option Type"]).strip().lower().startswith("c")
        S = float(o["Underlying"])
        K = float(o["Strike"])
        T = float(o["DaysToMaturity"]) / float(o["DayPerYear"])
        r = float(o["RiskFreeRate"])
        q = float(o["DividendRate"])
        b = r - q
        sigma = float(o["ImpliedVol"])

        parms = np.array([S, K, T, r, b, sigma], dtype=float)

        if is_call:
            v0 = fcall(parms)
            grad = finite_difference_gradient(fcall, parms)
        else:
            v0 = fput(parms)
            grad = finite_difference_gradient(fput, parms)

        deltas.append(float(grad[0]))

        dS = 1.5
        p_up = parms.copy(); p_dn = parms.copy()
        p_up[0] = parms[0] + dS
        p_dn[0] = parms[0] - dS
        v_up = fcall(p_up) if is_call else fput(p_up)
        v_dn = fcall(p_dn) if is_call else fput(p_dn)
        gamma = (v_up + v_dn - 2.0 * v0) / (dS ** 2)
        gammas.append(float(gamma))

        vegas.append(float(grad[5]))   # dV/dsigma
        rhos.append(float(grad[3]))    # dV/dr
        thetas.append(float(grad[2]))  # dV/dT

    save_output(options["ID"].to_numpy(),
                np.array(out_vals), np.array(deltas), np.array(gammas),
                np.array(vegas), np.array(rhos), np.array(thetas),
                out_path, decimals=6)
