"""Generate data/synthetic_peers_demo.csv — SIMULATED peer prices for the offline demo.

Each fake peer's daily return = beta * (target's same-day return) + random noise.
Peers move *with* the target on the same day but do not lead it, so the demo
contains no built-in predictive signal. This file is NOT real market data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
main = pd.read_csv(ROOT / "data" / "synthetic_demo.csv", parse_dates=["Date"])
ret = (main["Close"] / main["Close"].shift(1) - 1).fillna(0).to_numpy()
rng = np.random.default_rng(7)
out = pd.DataFrame({"Date": main["Date"].dt.strftime("%Y-%m-%d")})
for name, beta, noise, start in [("SYN_PARTS_A", 0.8, 0.012, 250.0), ("SYN_PARTS_B", 0.5, 0.015, 90.0),
                                 ("SYN_TYRES", 0.4, 0.014, 1200.0), ("SYN_STEEL", 0.6, 0.018, 140.0),
                                 ("SYN_INDEX", 0.7, 0.006, 18000.0)]:
    r = beta * ret + rng.normal(0, noise, len(ret))
    r[0] = 0
    out[name] = np.round(start * np.cumprod(1 + r), 4)
out.to_csv(ROOT / "data" / "synthetic_peers_demo.csv", index=False)
print("wrote", len(out), "rows")
