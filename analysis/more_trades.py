"""طرق لزيادة عدد الصفقات: مقارنة بمحاكاة حساب كامل (1% مخاطرة لكل صفقة)."""
import os

import numpy as np
import pandas as pd
import yfinance as yf

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "crypto_research.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(here, "crypto_research.py")}
exec(src.split("# ---------------- الشبكة ----------------")[0], ns)
D, SPLIT, atr_fn = ns["D"], ns["SPLIT"], ns["atr_fn"]
BASE = dict(D)

EXTRA_SYMS = {"NEAR": "NEAR-USD", "SUI": "SUI20947-USD", "INJ": "INJ-USD", "FIL": "FIL-USD", "ICP": "ICP-USD", "UNI": "UNI7083-USD"}
EXTRA = {}
for c, s in EXTRA_SYMS.items():
    h = yf.Ticker(s).history(period="max", interval="1d", auto_adjust=False)
    h.index = pd.to_datetime(h.index, utc=True)
    df = h.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})[["open", "high", "low", "close"]].astype(float).dropna()
    df["atr"] = atr_fn(df); df["sma50"] = df["close"].rolling(50).mean(); df["sma200"] = df["close"].rolling(200).mean()
    df["sma5"] = df["close"].rolling(5).mean(); df["rsi2"] = 50.0
    EXTRA[c] = df


def trades(coins, gen, tag):
    ns["D"].clear(); ns["D"].update(coins)
    rows = ns["run_config"](tag, gen)
    for x in rows:
        x["pool"] = tag
    ns["D"].clear(); ns["D"].update(BASE)
    return rows


def simulate(rows, caps, start, risk=0.01):
    rows = sorted([x for x in rows if x["t_in"] >= start], key=lambda x: (x["t_in"], x["coin"]))
    ev = sorted([(x["t_in"], 1, i) for i, x in enumerate(rows)] + [(x["t_out"], 0, i) for i, x in enumerate(rows)],
                key=lambda e: (e[0], e[1]))
    eq, peak, mdd, op, taken = 1.0, 1.0, 0.0, {}, 0
    for t, typ, i in ev:
        pool = rows[i]["pool"]
        if typ == 1:
            if sum(1 for j in op if rows[j]["pool"] == pool) < caps[pool]:
                op[i] = eq * risk; taken += 1
        elif i in op:
            eq += rows[i]["R"] * op.pop(i); peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    yrs = (pd.Timestamp.now(tz="UTC") - start).days / 365.25
    return (eq ** (1 / yrs) - 1) * 100, mdd * 100, taken / (yrs * 12)


ALL = {**BASE, **EXTRA}
d20 = lambda coins: trades(coins, ns["donchian_gen"](20, "donchian10", "btc>200"), "D")
R = {
    "d20_16": d20(BASE), "d20_22": d20(ALL),
    "d15_16": trades(BASE, ns["donchian_gen"](15, "donchian10", "btc>200"), "D"),
    "d10_16": trades(BASE, ns["donchian_gen"](10, "donchian5", "btc>200"), "D"),
    "d10x10_16": trades(BASE, ns["donchian_gen"](10, "donchian10", "btc>200"), "D"),
    "w3_16": trades(BASE, ns["w3_gen"](3.0, (0.382, 0.786), "w2", "tp2.618", "none"), "W"),
    "w3_22": trades(ALL, ns["w3_gen"](3.0, (0.382, 0.786), "w2", "tp2.618", "none"), "W"),
}
VARIANTS = [
    ("الحالي: اختراق 20، الـ16 عملة، حد 8", R["d20_16"], {"D": 8}),
    ("1أ) اختراق 15 يوم، حد 8", R["d15_16"], {"D": 8}),
    ("1ب) اختراق 10 يوم (خروج 5)، حد 8", R["d10_16"], {"D": 8}),
    ("1ج) اختراق 10 يوم (خروج 10)، حد 8", R["d10x10_16"], {"D": 8}),
    ("2) الحالي + إليوت بـ3 أماكن خاصة", R["d20_16"] + R["w3_16"], {"D": 8, "W": 3}),
    ("2ب) الحالي + إليوت بـ5 أماكن خاصة", R["d20_16"] + R["w3_16"], {"D": 8, "W": 5}),
    ("3) 22 عملة (الـ16 + 6)، حد 8", R["d20_22"], {"D": 8}),
    ("3ب) 22 عملة، حد 10", R["d20_22"], {"D": 10}),
    ("2+3) 22 عملة حد 10 + إليوت 3 أماكن", R["d20_22"] + R["w3_22"], {"D": 10, "W": 3}),
]
for lbl, start in (("من 2021 (فيه هبوط 2022)", pd.Timestamp("2021-01-01", tz="UTC")),
                   ("فترة الاختبار (منتصف 2023 → اليوم)", SPLIT),
                   ("آخر سنة ونص (2025 → اليوم، سوق ضعيف)", pd.Timestamp("2025-01-01", tz="UTC"))):
    print(f"\n######## {lbl} ########")
    for name, rows, caps in VARIANTS:
        cagr, mdd, tpm = simulate(rows, caps, start)
        print(f"  {name:40s} صفقات/شهر={tpm:4.1f} | سنوي={cagr:+6.1f}% | أقصى تراجع={mdd:5.1f}% | العائد÷التراجع={cagr / max(mdd, 0.1):4.2f}")
