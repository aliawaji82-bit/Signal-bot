"""اختبار عملات جديدة قبل إضافتها لبوت الكريبتو اليومي (نفس قواعد البوت بالضبط)."""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "crypto_research.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(here, "crypto_research.py")}
exec(src.split("# ---------------- الشبكة ----------------")[0], ns)
D, SPLIT, atr_fn = ns["D"], ns["SPLIT"], ns["atr_fn"]
BASE = dict(D)

# رمز Yahoo الأساسي + بدائل معروفة (بعض العملات لها رقم بالرمز)
CANDIDATES = {
    "NEAR": ["NEAR-USD"], "APT": ["APT21794-USD", "APT-USD"], "SUI": ["SUI20947-USD", "SUI-USD"],
    "ARB": ["ARB11841-USD", "ARB-USD"], "OP": ["OP-USD"], "INJ": ["INJ-USD"], "FIL": ["FIL-USD"],
    "ICP": ["ICP-USD"], "HBAR": ["HBAR-USD"], "VET": ["VET-USD"], "AAVE": ["AAVE-USD"],
    "UNI": ["UNI7083-USD", "UNI-USD"], "TON": ["TON11419-USD", "TON-USD"], "SHIB": ["SHIB-USD"],
    "PEPE": ["PEPE24478-USD", "PEPE-USD"], "RENDER": ["RENDER-USD", "RNDR-USD"], "ALGO": ["ALGO-USD"],
    "XMR": ["XMR-USD"], "SEI": ["SEI-USD"], "TIA": ["TIA22861-USD", "TIA-USD"], "KAS": ["KAS-USD"],
    "STX": ["STX4847-USD", "STX-USD"], "IMX": ["IMX10603-USD", "IMX-USD"], "GRT": ["GRT6719-USD", "GRT-USD"],
    "FET": ["FET-USD"], "LDO": ["LDO-USD"],
}
MIN_VOLUME = 50e6      # أقل متوسط تداول يومي (دولار) آخر 90 يوم
MIN_DAYS = 400


def prep(df):
    df = df.copy()
    df["atr"] = atr_fn(df)
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    d = df["close"].diff()
    g, l = d.clip(lower=0).rolling(2).mean(), (-d.clip(upper=0)).rolling(2).mean()
    df["rsi2"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    df["sma5"] = df["close"].rolling(5).mean()
    return df


NEW, INFO = {}, {}
print("=== البيانات والسيولة ===")
for coin, syms in CANDIDATES.items():
    got = None
    for s in syms:
        try:
            h = yf.Ticker(s).history(period="max", interval="1d", auto_adjust=False)
            if h is not None and len(h) > 50:
                got = (s, h)
                break
        except Exception:
            pass
    if not got:
        print(f"  {coin:7s} ❌ ما فيه بيانات على Yahoo ({', '.join(syms)})")
        continue
    s, h = got
    h.index = pd.to_datetime(h.index, utc=True)
    h = h[h.index >= "2017-06-01"]
    vol = (h["Close"] * h["Volume"]).tail(90).median()
    df = h.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})[["open", "high", "low", "close"]].astype(float).dropna()
    INFO[coin] = dict(sym=s, days=len(df), start=df.index[0], vol=vol)
    ok = len(df) >= MIN_DAYS and vol >= MIN_VOLUME
    print(f"  {coin:7s} {s:15s} {len(df):5d} يوم من {df.index[0]:%Y-%m-%d} | تداول يومي ≈ ${vol / 1e6:8,.0f}M {'✅' if ok else '⛔ ' + ('تاريخ قصير' if len(df) < MIN_DAYS else 'سيولة ضعيفة')}")
    if ok:
        NEW[coin] = prep(df)


GENS = {"Donchian20": ns["donchian_gen"](20, "donchian10", "btc>200"),
        "ElliottW3": ns["w3_gen"](3.0, (0.382, 0.786), "w2", "tp2.618", "none")}


def trades_for(coins_dfs):
    ns["D"].clear(); ns["D"].update(coins_dfs)
    rows = []
    for name, g in GENS.items():
        for x in ns["run_config"](name, g):
            x["strategy"] = name
            rows.append(x)
    ns["D"].clear(); ns["D"].update(BASE)
    return rows


def stats(rows):
    if not rows:
        return "لا صفقات"
    r = np.array([x["R"] for x in rows])
    c = np.array([ns["control"](NEW.get(x["coin"], BASE.get(x["coin"])), x["tr"], n=10) for x in rows])
    return (f"صفقات={len(r):3d} نجاح={(r > 0).mean() * 100:4.0f}% صافي={r.sum():+6.1f}R متوسط={r.mean():+.2f}R "
            f"| عشوائي={np.nanmean(c):+.2f}R ميزة={r.mean() - np.nanmean(c):+.2f}R"), r.sum(), r.mean() - np.nanmean(c), len(r)


print("\n=== النتيجة لكل عملة (نفس قواعد البوت) ===")
accepted = []
for coin, df in NEW.items():
    rows = trades_for({coin: df})
    line = [f"\n{coin} ({INFO[coin]['days']} يوم)"]
    total, verdict = 0.0, []
    for strat in GENS:
        rs = [x for x in rows if x["strategy"] == strat]
        st = stats(rs)
        if isinstance(st, str):
            line.append(f"   {strat:10s} {st}")
            continue
        txt, tot, edge, n = st
        total += tot
        verdict.append((strat, tot, edge, n))
        line.append(f"   {strat:10s} {txt}")
    d = next((v for v in verdict if v[0] == "Donchian20"), None)
    ok = d is not None and d[3] >= 8 and d[1] > 0 and d[2] > 0 and total > 0
    line.append(f"   ⇐ {'✅ نضيفها' if ok else '❌ ما نضيفها'} (المجموع {total:+.1f}R)")
    print("\n".join(line))
    if ok:
        accepted.append(coin)


def simulate(rows, start, risk=0.01, max_open=5):
    rows = sorted([x for x in rows if x["t_in"] >= start], key=lambda x: (x["t_in"], x["coin"]))
    ev = sorted([(x["t_in"], 1, i) for i, x in enumerate(rows)] + [(x["t_out"], 0, i) for i, x in enumerate(rows)],
                key=lambda e: (e[0], e[1]))
    eq, peak, mdd, op, taken, skipped = 1.0, 1.0, 0.0, {}, 0, 0
    for t, typ, i in ev:
        if typ == 1:
            if len(op) < max_open:
                op[i] = eq * risk; taken += 1
            else:
                skipped += 1
        elif i in op:
            eq += rows[i]["R"] * op.pop(i); peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    yrs = (pd.Timestamp.now(tz="UTC") - start).days / 365.25
    return (eq ** (1 / yrs) - 1) * 100, (eq - 1) * 100, mdd * 100, taken, skipped


print("\n=== محاكاة الحساب الكامل (الاستراتيجيتين، 1% مخاطرة، أقصى 5 صفقات) ===")
base_rows = trades_for(BASE)
new_rows = trades_for({c: NEW[c] for c in accepted}) if accepted else []
all_rows = trades_for({c: NEW[c] for c in NEW})
for lbl, start in (("من 2021", pd.Timestamp("2021-01-01", tz="UTC")), ("فترة الاختبار من منتصف 2023", SPLIT)):
    for name, rows, mo in (("الـ16 الحالية", base_rows, 5), ("الـ16 + المقبولة", base_rows + new_rows, 5),
                           ("الـ16 + المقبولة (حد 8 صفقات)", base_rows + new_rows, 8), ("الـ16 + كل الجديدة", base_rows + all_rows, 5)):
        cagr, tot, mdd, taken, skipped = simulate(rows, start, max_open=mo)
        print(f"  {lbl:28s} {name:30s} سنوي={cagr:+5.1f}% كلي={tot:+6.0f}% أقصى تراجع={mdd:4.1f}% صفقات={taken} فاتت بسبب الحد={skipped}")
print(f"\nالعملات المقبولة: {accepted}")
print("رموز Yahoo: " + str({c: INFO[c]['sym'] for c in accepted}))
