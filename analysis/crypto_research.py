"""
بحث شامل على العملات الرقمية (فريم يومي، 16 عملة، 2018 → اليوم).
- تدريب: 2018-01-01 → 2023-06-30 (فيه سوقين هابطين 2018 و2022) | اختبار: 2023-07-01 → اليوم.
- نختار الأفضل من التدريب فقط، ونشوف نتيجته بالاختبار.
- لكل استراتيجية: مقارنة بدخول عشوائي بنفس الاتجاه ونفس قواعد الخروج.
- محاكاة حساب فعلي بمخاطرة 1% لكل صفقة للأفضل.
"""
import itertools
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_patterns.py"), encoding="utf-8").read()
ns = {}
exec(src.split('print("تحميل البيانات...")')[0], ns)
fetch, zigzag, atr_fn, wait_fill, COINS = ns["fetch"], ns["zigzag"], ns["atr"], ns["wait_fill"], ns["COINS"]

FEE = 0.002
SPLIT = pd.Timestamp("2023-07-01", tz="UTC")
RISK = 0.01
random.seed(11)

# ---------------- البيانات ----------------
D = {}
for c in COINS:
    try:
        df = fetch(c, "1d", "max")
        df = df[df.index >= "2017-06-01"].copy()
        df["atr"] = atr_fn(df)
        df["sma50"] = df["close"].rolling(50).mean()
        df["sma200"] = df["close"].rolling(200).mean()
        d = df["close"].diff()
        g, l = d.clip(lower=0).rolling(2).mean(), (-d.clip(upper=0)).rolling(2).mean()
        df["rsi2"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
        df["sma5"] = df["close"].rolling(5).mean()
        D[c] = df
        print(f"{c}: {len(df)} يوم من {df.index[0]:%Y-%m-%d}")
    except Exception as e:
        print(f"{c}: فشل {e}")
BTC_UP = (D["BTC"]["close"] > D["BTC"]["sma200"])


def btc_up(ts):
    s = BTC_UP[BTC_UP.index <= ts]
    return bool(len(s) and s.iloc[-1])


# ---------------- محرّك الخروج ----------------
def exit_trade(df, j, entry, stop, mode, target=None, trail_mult=3.0, max_hold=250, check_fill_bar=True):
    """شراء فقط. mode: 'target' | 'trail' (Chandelier) | 'mr' (خروج فوق SMA5 أو بعد 10 أيام)
       | 'donchian10' / 'donchian20' (خروج عند كسر أدنى قاع N يوم). يرجع (R, exit_idx) أو None."""
    hi, lo, cl, op, a, sma5 = (df[k].values for k in ("high", "low", "close", "open", "atr", "sma5"))
    risk = entry - stop
    if risk <= 0 or risk / entry < 0.003:
        return None
    fee_r = FEE * entry / risk
    st, hh = stop, entry
    end = min(len(df), j + max_hold)
    for t in range(j, end):
        if lo[t] <= st:
            px = min(st, op[t]) if t > j else st
            return (px - entry) / risk - fee_r, t
        if t == j and check_fill_bar:
            continue
        if mode == "target" and hi[t] >= target:
            return (target - entry) / risk - fee_r, t
        if mode == "trail":
            hh = max(hh, hi[t])
            st = max(st, hh - trail_mult * a[t])
        if mode == "mr" and (cl[t] > sma5[t] or t - j >= 10):
            return (cl[t] - entry) / risk - fee_r, t
        if mode.startswith("donchian"):
            n = int(mode[8:])
            if t - n >= 0 and cl[t] < lo[t - n:t].min():
                return (cl[t] - entry) / risk - fee_r, t
    if end >= len(df):
        return None
    return (cl[end - 1] - entry) / risk - fee_r, end - 1


def control(df, trade, n=20):
    """نفس قواعد الخروج ونفس مسافة الوقف النسبية، دخول بسعر الإغلاق بيوم عشوائي."""
    out = []
    sd = (trade["entry"] - trade["stop"]) / trade["entry"]
    td = None if trade.get("target") is None else (trade["target"] - trade["entry"]) / trade["entry"]
    cl = df["close"].values
    lo_i = 210
    for _ in range(n):
        k = random.randrange(lo_i, len(df) - 30)
        e = cl[k]
        r = exit_trade(df, k + 1, e, e * (1 - sd), trade["mode"], None if td is None else e * (1 + td),
                       trade.get("trail", 3.0), check_fill_bar=False)
        if r:
            out.append(r[0])
    return np.mean(out) if out else np.nan


# ---------------- مولّدات الصفقات ----------------
def w3_candidates(df, k, retr):
    """نقاط دخول موجة 3 (شراء): يرجع dict فيها j وسعر الدخول وقاع 2 وبداية 1 وطول 1."""
    piv = zigzag(df, k)
    out = []
    for m in range(2, len(piv)):
        (i0, p0, _, _), (i1, p1, _, _), (i2, p2, t2, c2) = piv[m - 2: m + 1]
        if t2 != "L" or p2 <= p0:
            continue
        w1 = p1 - p0
        if w1 <= 0 or not (retr[0] <= (p1 - p2) / w1 <= retr[1]):
            continue
        nxt = piv[m + 1][3] if m + 1 < len(piv) else 10**9
        f = wait_fill(df, c2 + 1, "BUY", "stop", p1, cancel_below=p2, max_wait=min(nxt - c2, 60))
        if f:
            out.append(dict(j=f[0], entry=f[1], p0=p0, p2=p2, w1=w1))
    return out


def filt_ok(df, j, fname, ts):
    if fname == "none":
        return True
    r = df.iloc[j - 1]
    if fname == "coin>200":
        return r["close"] > r["sma200"]
    if fname == "50>200":
        return r["sma50"] > r["sma200"]
    if fname == "btc>200":
        return btc_up(df.index[j - 1])
    if fname == "btc&coin>200":
        return btc_up(df.index[j - 1]) and r["close"] > r["sma200"]
    return True


def run_config(name, gen):
    """gen(coin, df) → قائمة صفقات dict(j, entry, stop, mode, target, trail). يرجع صفوف بالنتائج."""
    rows = []
    for coin, df in D.items():
        for tr in gen(coin, df):
            if tr["j"] < 210 or tr["j"] >= len(df):
                continue
            r = exit_trade(df, tr["j"], tr["entry"], tr["stop"], tr["mode"], tr.get("target"), tr.get("trail", 3.0),
                           check_fill_bar=tr.get("fillcheck", True))
            if r is None:
                continue
            ts = df.index[tr["j"]]
            rows.append(dict(cfg=name, coin=coin, R=r[0], t_in=ts, t_out=df.index[r[1]], tr=tr,
                             ctrl=np.nan, part="test" if ts >= SPLIT else "train"))
    return rows


def w3_gen(k, retr, stop_mode, exit_mode, filt):
    def g(coin, df):
        out = []
        for c in w3_candidates_cached(coin, df, k, retr):
            if not filt_ok(df, c["j"], filt, None):
                continue
            stop = c["p2"] if stop_mode == "w2" else c["p0"]
            tr = dict(j=c["j"], entry=c["entry"], stop=stop)
            if exit_mode.startswith("tp"):
                mult = float(exit_mode[2:])
                tr.update(mode="target", target=c["p2"] + mult * c["w1"])
                if tr["target"] <= tr["entry"]:
                    continue
            else:
                tr.update(mode="trail", trail=float(exit_mode[5:]))
            out.append(tr)
        return out
    return g


_cache = {}
def w3_candidates_cached(coin, df, k, retr):
    key = (coin, k, retr)
    if key not in _cache:
        _cache[key] = w3_candidates(df, k, retr)
    return _cache[key]


def donchian_gen(n_in, exit_mode, filt):
    def g(coin, df):
        out, hi, cl, op, a = [], df["high"].values, df["close"].values, df["open"].values, df["atr"].values
        pos_until = -1
        for t in range(n_in + 1, len(df) - 1):
            if t <= pos_until:
                continue
            if cl[t] > hi[t - n_in:t].max() and filt_ok(df, t + 1, filt, None):
                e = op[t + 1]
                tr = dict(j=t + 1, entry=e, stop=e - 2 * a[t], mode=exit_mode, fillcheck=False)
                if exit_mode.startswith("trail"):
                    tr.update(mode="trail", trail=float(exit_mode[5:]))
                r = exit_trade(df, tr["j"], tr["entry"], tr["stop"], tr["mode"], None, tr.get("trail", 3.0), check_fill_bar=False)
                pos_until = r[1] if r else len(df)
                out.append(tr)
        return out
    return g


def mr_gen(rsi_th, filt):
    def g(coin, df):
        out, cl, op, a, rsi = [], df["close"].values, df["open"].values, df["atr"].values, df["rsi2"].values
        pos_until = -1
        for t in range(210, len(df) - 1):
            if t <= pos_until or not (rsi[t] < rsi_th) or not filt_ok(df, t + 1, filt, None):
                continue
            e = op[t + 1]
            tr = dict(j=t + 1, entry=e, stop=e - 2.5 * a[t], mode="mr", fillcheck=False)
            r = exit_trade(df, tr["j"], e, tr["stop"], "mr", check_fill_bar=False)
            pos_until = r[1] if r else len(df)
            out.append(tr)
        return out
    return g


# ---------------- الشبكة ----------------
CONFIGS = {}
for k, retr, sm, ex, f in itertools.product((2.0, 2.5, 3.0, 4.0), ((0.382, 0.786), (0.5, 0.786)), ("w2", "w1start"),
                                            ("tp1.0", "tp1.618", "tp2.618", "trail3.0", "trail4.0"),
                                            ("none", "coin>200", "50>200", "btc>200", "btc&coin>200")):
    CONFIGS[f"W3 k{k} r{retr[0]} stop:{sm} {ex} f:{f}"] = w3_gen(k, retr, sm, ex, f)
for n, ex, f in itertools.product((20, 55), ("donchian10", "donchian20", "trail3.0", "trail4.0"), ("none", "btc>200", "coin>200")):
    CONFIGS[f"Donchian{n} {ex} f:{f}"] = donchian_gen(n, ex, f)
for th, f in itertools.product((5, 10), ("coin>200", "btc&coin>200")):
    CONFIGS[f"RSI2<{th} MR f:{f}"] = mr_gen(th, f)

print(f"\nعدد الإعدادات المختبرة: {len(CONFIGS)}")
RES = {}
for name, gen in CONFIGS.items():
    RES[name] = run_config(name, gen)


def add_control(name):
    for x in RES[name]:
        if np.isnan(x["ctrl"]):
            x["ctrl"] = control(D[x["coin"]], x["tr"], n=15)


def st(rows):
    if not rows:
        return None
    r = np.array([x["R"] for x in rows]); c = np.array([x["ctrl"] for x in rows])
    pf = r[r > 0].sum() / max(1e-9, -r[r < 0].sum())
    ctrl = np.nanmean(c) if np.isfinite(c).any() else np.nan
    return dict(n=len(r), wr=(r > 0).mean() * 100, tot=r.sum(), avg=r.mean(), pf=pf, ctrl=ctrl, edge=r.mean() - ctrl)


def fmt(s):
    if s is None:
        return "لا صفقات"
    return (f"صفقات={s['n']:4d} نجاح={s['wr']:5.1f}% صافي={s['tot']:+7.1f}R متوسط={s['avg']:+.3f}R PF={s['pf']:4.2f} "
            f"عشوائي={s['ctrl']:+.3f} ميزة={s['edge']:+.3f}")


table = []
for name, rows in RES.items():
    tr = st([x for x in rows if x["part"] == "train"]); te = st([x for x in rows if x["part"] == "test"])
    if tr and tr["n"] >= 40:
        table.append((name, tr, te))

# الترتيب حسب متوسط R بالتدريب - الاختيار من التدريب فقط. بعدين نحسب المقارنة العشوائية للأفضل
table.sort(key=lambda x: x[1]["avg"], reverse=True)
fam_best = []
for fam in ("W3", "Donchian20", "Donchian55", "RSI2"):
    fam_best += [t for t in table if t[0].startswith(fam)][:1]
for name, _, _ in table[:25] + fam_best:
    add_control(name)
table = [(n, st([x for x in RES[n] if x["part"] == "train"]), st([x for x in RES[n] if x["part"] == "test"])) for n, _, _ in table]
print("\n######## أفضل 25 إعداد حسب فترة التدريب، ونتيجتهم بالاختبار ########")
for name, tr, te in table[:25]:
    print(f"\n{name}\n   تدريب : {fmt(tr)}\n   اختبار: {fmt(te)}")

print("\n######## ملخص العائلات (أفضل إعداد بالتدريب لكل عائلة) ########")
for fam in ("W3", "Donchian20", "Donchian55", "RSI2"):
    cand = [t for t in table if t[0].startswith(fam)]
    if cand:
        name, tr, te = cand[0]
        print(f"\n[{fam}] {name}\n   تدريب : {fmt(tr)}\n   اختبار: {fmt(te)}")

print("\n######## أثر الفلاتر على W3 (متوسط كل إعدادات W3 لكل فلتر) ########")
for f in ("none", "coin>200", "50>200", "btc>200", "btc&coin>200"):
    for part in ("train", "test"):
        rows = [x for n, rs in RES.items() if n.startswith("W3") and n.endswith(f"f:{f}") for x in rs if x["part"] == part]
        print(f"  {f:14s} {part:5s} {fmt(st(rows)).split(' عشوائي')[0]}")
print("\n######## أثر طريقة الخروج على W3 ########")
for ex in ("tp1.0", "tp1.618", "tp2.618", "trail3.0", "trail4.0"):
    for part in ("train", "test"):
        rows = [x for n, rs in RES.items() if n.startswith("W3") and f" {ex} " in n for x in rs if x["part"] == part]
        print(f"  {ex:9s} {part:5s} {fmt(st(rows)).split(' عشوائي')[0]}")


# ---------------- محاكاة حساب ----------------
def simulate(rows, start, risk=RISK):
    rows = sorted([x for x in rows if x["t_in"] >= start], key=lambda x: x["t_in"])
    ev = [(x["t_in"], 0, i) for i, x in enumerate(rows)] + [(x["t_out"], 1, i) for i, x in enumerate(rows)]
    ev.sort(key=lambda e: (e[0], e[1]))
    eq, peak, mdd, open_risk, maxc, cur = 1.0, 1.0, 0.0, {}, 0, 0
    for t, typ, i in ev:
        if typ == 0:
            open_risk[i] = eq * risk; cur += 1; maxc = max(maxc, cur)
        else:
            eq += rows[i]["R"] * open_risk.pop(i); cur -= 1
            peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    years = max(1e-9, (pd.Timestamp.now(tz="UTC") - start).days / 365.25)
    return eq, mdd, len(rows) / years, maxc, years


btc = D["BTC"]["close"]
bh = btc.iloc[-1] / btc[btc.index >= SPLIT].iloc[0] - 1
print(f"\n######## محاكاة حساب بالاختبار ({SPLIT:%Y-%m-%d} → اليوم) بمخاطرة 1% لكل صفقة ########")
print(f"للمقارنة: شراء BTC والاحتفاظ فيه بنفس الفترة = {bh * 100:+.0f}%")
seen = set()
for name, tr, te in table[:10] + [t for fam in ("Donchian20", "Donchian55", "RSI2") for t in [x for x in table if x[0].startswith(fam)][:1]]:
    if name in seen:
        continue
    seen.add(name)
    eq, mdd, tpy, maxc, yrs = simulate(RES[name], SPLIT)
    print(f"  {name:55s} العائد={(eq - 1) * 100:+6.1f}% | أقصى تراجع={mdd * 100:5.1f}% | صفقات/سنة={tpy:5.1f} | أقصى صفقات متزامنة={maxc}")
