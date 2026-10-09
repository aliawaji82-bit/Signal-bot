"""
اختبار تاريخي لقواعد Signal-bot وتعديلات عليها.
- البيانات: Yahoo. فريم 5د/15د (آخر 60 يوم)، وفريم 1س/4س (آخر سنتين).
- الفترة تنقسم: أول الثلثين "تدريب" نختار منه، والثلث الأخير "اختبار" ما شافه الاختيار.
- تكلفة تقريبية لكل صفقة (سبريد/عمولة) تنخصم من النتيجة.
- فلتر الأخبار ما يدخل بالاختبار (ما فيه تقويم تاريخي).
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "x")
os.environ.setdefault("TELEGRAM_CHAT_ID", "x")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scanner  # noqa: E402

COST_PCT = {"forex": 0.00010, "commodity": 0.00030, "crypto": 0.00080, "stock": 0.00030}
SESSION = {"forex": (7, 21), "commodity": (7, 21), "stock": (13, 20), "crypto": None}


def fetch(sym, interval, period):
    df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    return df[["open", "high", "low", "close"]].astype(float).dropna()


def resample_4h(df):
    return df.resample("4h", origin="epoch").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def prepare(entry, trend, entry_min, trend_min):
    e = scanner.add_indicators(entry.copy())
    t = scanner.add_indicators(trend.copy())
    e["avail"] = e.index + pd.Timedelta(minutes=entry_min)   # وقت إقفال الشمعة
    t["avail"] = t.index + pd.Timedelta(minutes=trend_min)
    t = t[["avail", "ema_fast", "ema_slow"]].rename(columns={"ema_fast": "t_fast", "ema_slow": "t_slow"})
    e = pd.merge_asof(e.reset_index().rename(columns={e.index.name or "index": "time"}).sort_values("avail"),
                      t.sort_values("avail"), on="avail", direction="backward")
    m, s = e["macd"].values, e["macd_signal"].values
    up = np.r_[False, (m[:-1] <= s[:-1]) & (m[1:] > s[1:])]
    dn = np.r_[False, (m[:-1] >= s[:-1]) & (m[1:] < s[1:])]
    # تقاطع خلال آخر 3 شموع (مثل macd_cross_recent)
    e["cross_up"] = pd.Series(up).rolling(3, min_periods=1).max().astype(bool).values
    e["cross_dn"] = pd.Series(dn).rolling(3, min_periods=1).max().astype(bool).values
    e["rsi_prev"] = e["rsi"].shift(1)
    e["res"] = e["high"].shift(1).rolling(scanner.SR_LOOKBACK).max()
    e["sup"] = e["low"].shift(1).rolling(scanner.SR_LOOKBACK).min()
    return e


def simulate(e, cat, p, split_time):
    """يرجع قائمة صفقات (وقت، نتيجة R، تدريب/اختبار)."""
    trades = []
    last_sig = {}
    hi, lo, cl = e["high"].values, e["low"].values, e["close"].values
    scan_min = p["scan_min"]
    sess = SESSION[cat] if p.get("session", True) else None
    cost_pct = COST_PCT[cat]
    av = e["avail"]
    mod = (av.dt.hour * 60 + av.dt.minute).values
    hour, wd = av.dt.hour.values, av.dt.weekday.values
    A = {k: e[k].values for k in ("close", "atr", "t_fast", "t_slow", "rsi", "rsi_prev", "cross_up", "cross_dn", "res", "sup")}
    for i in range(60, len(e) - 1):
        if mod[i] % scan_min != 0:
            continue
        if sess is not None and (wd[i] >= 5 or not (sess[0] <= hour[i] < sess[1])):
            continue
        row = {k: v[i] for k, v in A.items()}
        tclose = av.iloc[i]
        price, atr = row["close"], row["atr"]
        if not price or pd.isna(atr) or pd.isna(row["t_fast"]) or pd.isna(row["rsi_prev"]):
            continue
        if atr / price * 100 < scanner.MIN_ATR_PCT:
            continue
        up_tr, dn_tr = row["t_fast"] > row["t_slow"], row["t_fast"] < row["t_slow"]
        rsi_up = 25 < row["rsi"] < 60 and row["rsi"] <= row["rsi_prev"]
        rsi_dn = 40 < row["rsi"] < 75 and row["rsi"] >= row["rsi_prev"]
        if p["logic"] == "or":
            buy, sell = up_tr and (row["cross_up"] or rsi_up), dn_tr and (row["cross_dn"] or rsi_dn)
        elif p["logic"] == "and":
            buy, sell = up_tr and row["cross_up"] and 30 < row["rsi"] < 65, dn_tr and row["cross_dn"] and 35 < row["rsi"] < 70
        else:  # macd فقط
            buy, sell = up_tr and row["cross_up"], dn_tr and row["cross_dn"]
        if buy and (row["res"] - price) < scanner.SR_MIN_DISTANCE_ATR * atr:
            buy = False
        if sell and (price - row["sup"]) < scanner.SR_MIN_DISTANCE_ATR * atr:
            sell = False
        for d, ok in (("BUY", buy), ("SELL", sell)):
            if not ok:
                continue
            if d in last_sig and (tclose - last_sig[d]).total_seconds() < p["cooldown_min"] * 60:
                continue
            last_sig[d] = tclose
            sl_d, tp_d = atr * p["sl"], atr * p["tp"]
            cost = price * cost_pct
            res = None
            for j in range(i + 1, min(len(e), i + 1 + p["max_bars"])):
                if d == "BUY":
                    hit_sl, hit_tp = lo[j] <= price - sl_d, hi[j] >= price + tp_d
                else:
                    hit_sl, hit_tp = hi[j] >= price + sl_d, lo[j] <= price - tp_d
                if hit_sl:
                    res = -(sl_d + cost) / sl_d
                    break
                if hit_tp:
                    res = (tp_d - cost) / sl_d
                    break
            if res is None:  # انتهت المدة: نقفل بسعر الإغلاق
                j = min(len(e) - 1, i + p["max_bars"])
                if j >= len(e) - 1 and j - i < p["max_bars"]:
                    continue  # لسا مفتوحة بنهاية البيانات
                pnl = (cl[j] - price) if d == "BUY" else (price - cl[j])
                res = (pnl - cost) / sl_d
            trades.append((tclose, res, "train" if tclose < split_time else "test"))
    return trades


def stats(rs):
    if not rs:
        return "لا صفقات"
    rs = np.array(rs)
    wins = (rs > 0).mean() * 100
    pf = rs[rs > 0].sum() / max(1e-9, -rs[rs < 0].sum())
    return f"صفقات={len(rs):5d} | نجاح={wins:5.1f}% | صافي={rs.sum():+8.1f}R | متوسط={rs.mean():+.3f}R | PF={pf:.2f}"


VARIANTS = {
    # --- فريم 5د/15د (نفس البوت) ---
    "A الحالي (فحص كل 30د)":            dict(tf="5m", logic="or", sl=1.2, tp=2.4, scan_min=30, cooldown_min=30, max_bars=288),
    "B الحالي بالتأخير الفعلي (كل 6س)":  dict(tf="5m", logic="or", sl=1.2, tp=2.4, scan_min=360, cooldown_min=30, max_bars=288),
    "C شرط MACD و RSI مع بعض":           dict(tf="5m", logic="and", sl=1.2, tp=2.4, scan_min=30, cooldown_min=30, max_bars=288),
    "D تقاطع MACD بس":                   dict(tf="5m", logic="macd", sl=1.2, tp=2.4, scan_min=30, cooldown_min=30, max_bars=288),
    "E وقف أوسع 2ATR وهدف 4ATR":         dict(tf="5m", logic="or", sl=2.0, tp=4.0, scan_min=30, cooldown_min=30, max_bars=288),
    "F هدف=وقف (1:1)":                   dict(tf="5m", logic="or", sl=1.5, tp=1.5, scan_min=30, cooldown_min=30, max_bars=288),
    "G MACD بس + وقف أوسع":              dict(tf="5m", logic="macd", sl=2.0, tp=4.0, scan_min=30, cooldown_min=60, max_bars=288),
    # --- فريم أبطأ 1س/4س (يناسب التشغيل كل كم ساعة) - بيانات سنتين ---
    "H فريم 1س/4س نفس القواعد":          dict(tf="1h", logic="or", sl=1.2, tp=2.4, scan_min=60, cooldown_min=240, max_bars=120),
    "I فريم 1س/4س MACD و RSI":           dict(tf="1h", logic="and", sl=1.5, tp=3.0, scan_min=60, cooldown_min=240, max_bars=120),
    "J فريم 1س/4س MACD بس، وقف 2":       dict(tf="1h", logic="macd", sl=2.0, tp=4.0, scan_min=60, cooldown_min=240, max_bars=120),
    "K فريم 1س/4س MACD بس، وقف 2، كل 6س": dict(tf="1h", logic="macd", sl=2.0, tp=4.0, scan_min=360, cooldown_min=240, max_bars=120),
}

data = {}
for c in scanner.SYMBOLS:
    try:
        e5, t15 = fetch(c["yf"], "5m", "60d"), fetch(c["yf"], "15m", "60d")
        h1 = fetch(c["yf"], "1h", "730d")
        data[c["label"]] = (c["category"], e5, t15, h1)
        print(f"{c['label']}: 5m={0 if e5 is None else len(e5)} 1h={0 if h1 is None else len(h1)}")
    except Exception as ex:
        print(f"{c['label']}: فشل {ex}")

prepared = {}
for label, (cat, e5, t15, h1) in data.items():
    try:
        if e5 is not None and t15 is not None and len(e5) > 300:
            prepared[(label, "5m")] = prepare(e5, t15, 5, 15)
        if h1 is not None and len(h1) > 300:
            prepared[(label, "1h")] = prepare(h1, resample_4h(h1), 60, 240)
    except Exception as ex:
        print(f"{label}: فشل التجهيز {ex}")

results = {}
for name, p in VARIANTS.items():
    by_cat = defaultdict(lambda: {"train": [], "test": []})
    for (label, tf), e in prepared.items():
        if tf != p["tf"]:
            continue
        cat = data[label][0]
        t0, t1 = e["avail"].iloc[60], e["avail"].iloc[-1]
        split = t0 + (t1 - t0) * 2 / 3
        for _, r, part in simulate(e, cat, p, split):
            by_cat[cat][part].append(r)
            by_cat["ALL"][part].append(r)
    results[name] = by_cat

print("\n" + "=" * 100)
print("النتائج (بعد خصم تكلفة تقريبية). PF = مجموع الأرباح ÷ مجموع الخسائر (فوق 1 = رابح)")
for part, title in (("train", "فترة التدريب (أول الثلثين)"), ("test", "فترة الاختبار (الثلث الأخير - ما شافها الاختيار)")):
    print(f"\n######## {title} ########")
    for name, by_cat in results.items():
        print(f"\n{name}")
        for cat in ("ALL", "forex", "commodity", "crypto", "stock"):
            if by_cat[cat][part]:
                print(f"   {cat:9s} {stats(by_cat[cat][part])}")
