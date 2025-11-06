import numpy as np
import pandas as pd


def american_discrete_div_price(
    is_call: bool,
    S0: float, K: float,
    T_years: float, r: float, sigma: float,
    steps: int,
    div_steps: list[int],      
    div_amts: list[float],   
) -> float:
    """
    CRR American option with *discrete cash dividends*.
    At each ex-div step i, continuation is evaluated from post-div stock S' = S - D
    using linear interpolation on the (i+1) stock grid. Early exercise at all steps.
    """
    if S0 <= 0 or K <= 0 or T_years <= 0 or sigma <= 0 or steps < 2:
        return max(S0 - K, 0.0) if is_call else max(K - S0, 0.0)

    dt = T_years / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    # Risk-neutral prob with no continuous yield (dividends are discrete jumps)
    p = (np.exp(r * dt) - d) / (u - d)
    if p < 0.0 or p > 1.0:
        return float("nan")
    disc = np.exp(-r * dt)

    # Precompute stock grids S[i][j] = S0 * u^j * d^(i-j)
    S_grid = [None] * (steps + 1)
    for i in range(steps + 1):
        j = np.arange(i + 1)
        S_grid[i] = S0 * (u ** j) * (d ** (i - j))

    # Terminal payoff
    ST = S_grid[steps]
    V = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)

    div_map = {int(s): float(a) for s, a in zip(div_steps, div_amts)}

    def lin_interp(xgrid: np.ndarray, ygrid: np.ndarray, x: float) -> float:
        """Monotone 1D linear interpolation with edge clamping by linear extrapolation."""
        if x <= xgrid[0]:
            x0, x1 = xgrid[0], xgrid[1]; y0, y1 = ygrid[0], ygrid[1]
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        if x >= xgrid[-1]:
            x0, x1 = xgrid[-2], xgrid[-1]; y0, y1 = ygrid[-2], ygrid[-1]
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        lo, hi = 0, len(xgrid) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xgrid[mid] <= x: lo = mid
            else: hi = mid
        x0, x1 = xgrid[lo], xgrid[hi]; y0, y1 = ygrid[lo], ygrid[hi]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    # Backward induction with dividend jumps and early exercise
    for i in range(steps - 1, -1, -1):
        V_std = disc * (p * V[1:i + 2] + (1.0 - p) * V[0:i + 1])

        if i in div_map:
            D = div_map[i]
            S_before = S_grid[i]
            S_after = np.maximum(S_before - D, 1e-12)   # avoid negative/zero post-div stock
            S_next = S_grid[i + 1]
            V_up = np.array([lin_interp(S_next, V, s * u) for s in S_after])
            V_dn = np.array([lin_interp(S_next, V, s * d) for s in S_after])
            V_new = disc * (p * V_up + (1.0 - p) * V_dn)
        else:
            V_new = V_std

        exercise = np.maximum(S_grid[i] - K, 0.0) if is_call else np.maximum(K - S_grid[i], 0.0)
        V = np.maximum(V_new, exercise)

    return float(V[0])


def load_input(path: str) -> pd.DataFrame:
    """Read input, keep non-missing IDs, parse dividend lists."""
    df = pd.read_csv(path, dtype={"ID": "Int64"})
    df = df.dropna(subset=["ID"])
    df["DividendDates"] = df["DividendDates"].astype(str).apply(
        lambda s: [int(t) for t in s.split(",") if t.strip() != ""]
    )
    df["DividendAmts"] = df["DividendAmts"].astype(str).apply(
        lambda s: [float(t) for t in s.split(",") if t.strip() != ""]
    )
    return df


def save_output_price(ids, values, path: str, decimals: int = 6):
    out = pd.DataFrame({"ID": ids, "Value": values})
    out["ID"] = pd.Series(out["ID"], dtype="Int64")
    out = out.round(decimals)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    in_path = "test12_3.csv"
    out_path = "testout12_3.csv"

    df = load_input(in_path)

    df["N"] = (df["DaysToMaturity"] * 2).astype(int)
    df["DividendDates"] = df.apply(
        lambda row: [int(d * 2) for d in row["DividendDates"]],
        axis=1
    )

    values = []
    for _, o in df.iterrows():
        is_call = str(o["Option Type"]).strip().lower().startswith("c")
        S0 = float(o["Underlying"])
        K = float(o["Strike"])
        T_years = float(o["DaysToMaturity"]) / float(o["DayPerYear"])
        r = float(o["RiskFreeRate"])
        sigma = float(o["ImpliedVol"])
        N = int(o["N"])
        div_steps = list(o["DividendDates"])
        div_amts = list(o["DividendAmts"])

        price = american_discrete_div_price(
            is_call, S0, K, T_years, r, sigma, N, div_steps, div_amts
        )
        values.append(price)

    save_output_price(df["ID"].to_numpy(), np.array(values), out_path, decimals=6)
