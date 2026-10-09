"""
اختبار موجات إليوت والأنماط التوافقية (Harmonic) على العملات الرقمية.

كل شي مبني على ZigZag سببي: القمة/القاع ما تنعتمد إلا بعد ما يرتد السعر عنها
بمقدار (k × ATR) - يعني ما فيه نظر للمستقبل.

إليوت:
  W3  : بعد موجة 1 و2 (التصحيح 38.2%-78.6% وما ينزل تحت بداية 1) → أمر دخول عند كسر قمة 1،
        وقف تحت قاع 2، هدف = قاع 2 + 1.618×موجة 1
  W5  : بعد موجات 1-4 بقواعد إليوت (2 ما تكسر بداية 1، 4 ما تتداخل مع 1، 3 مو الأقصر بين 1 و3،
        تصحيح 4 بين 23.6%-61.8% من 3) → دخول عند تأكيد قاع 4، وقف تحت قاع 4، هدف = موجة 5 تساوي موجة 1
  REV : بعد موجة دافعة كاملة 1-5 بالقواعد → صفقة عكسية عند تأكيد قمة 5، وقف فوقها، هدف تصحيح 38.2% من 0→5
هارمونك (XABCD): Gartley / Bat / Butterfly / Crab / Cypher بالنسب القياسية،
  أمر محدد (Limit) عند نقطة D، وقف خلف X (أو خلف امتداد D للفراشة والسلطعون)، هدف 38.2% أو 61.8% من AD.

لكل صفقة نحسب صفقات "عشوائية" بنفس الاتجاه ونفس مسافة الوقف والهدف بأوقات عشوائية، عشان نعرف
هل النمط يعطي ميزة حقيقية أو النتيجة جاية من اتجاه السوق العام.
"""
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC", "BCH", "TRX", "XLM", "ATOM", "ETC"]
FEE = 0.002          # عمولة ذهاب وإياب تقريبية (0.1% لكل جهة)
MAX_HOLD = 60        # أقصى مدة للصفقة بالشموع
RANDOM_SAMPLES = 30
random.seed(7)


def fetch(sym, interval, period):
    df = yf.Ticker(f"{sym}-USD").history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})[["open", "high", "low", "close"]]
    return df.astype(float).dropna()


def to_4h(df):
    return df.resample("4h", origin="epoch").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n).mean().values


def zigzag(df, k):
    """يرجع قائمة نقاط محورية: (idx, price, 'H'/'L', confirm_idx)."""
    hi, lo, a = df["high"].values, df["low"].values, atr(df)
    piv, d, ext, ext_i = [], 0, None, None
    for i in range(len(df)):
        if np.isnan(a[i]):
            continue
        thr = k * a[i]
        if d == 0:
            d, ext, ext_i = 1, hi[i], i
            continue
        if d == 1:
            if hi[i] >= ext:
                ext, ext_i = hi[i], i
            elif ext - lo[i] >= thr:
                piv.append((ext_i, ext, "H", i))
                d, ext, ext_i = -1, lo[i], i
        else:
            if lo[i] <= ext:
                ext, ext_i = lo[i], i
            elif hi[i] - ext >= thr:
                piv.append((ext_i, ext, "L", i))
                d, ext, ext_i = 1, hi[i], i
    return piv


def run_trade(df, j, direction, entry, stop, target, fill_bar_check_stop=True):
    """يمشي من شمعة الدخول j. يرجع R بعد العمولة أو None لو ما انقفلت قبل نهاية البيانات."""
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    risk = abs(entry - stop)
    if risk <= 0 or risk / entry < 0.002:
        return None
    fee_r = FEE * entry / risk
    buy = direction == "BUY"
    end = min(len(df), j + MAX_HOLD)
    for t in range(j, end):
        if (lo[t] <= stop) if buy else (hi[t] >= stop):
            return -1 - fee_r
        if t == j and fill_bar_check_stop:
            continue  # شمعة التنفيذ: نفحص الوقف بس (تحفّظ)
        if (hi[t] >= target) if buy else (lo[t] <= target):
            return abs(target - entry) / risk - fee_r
    if end >= len(df):
        return None
    px = cl[end - 1]
    return ((px - entry) if buy else (entry - px)) / risk - fee_r


def wait_fill(df, start, direction, kind, level, cancel_above=None, cancel_below=None, max_wait=40):
    """ينتظر تنفيذ أمر معلّق. kind='stop' (اختراق) أو 'limit' (ارتداد). يرجع (idx, سعر التنفيذ) أو None."""
    op, hi, lo = df["open"].values, df["high"].values, df["low"].values
    for t in range(start, min(len(df), start + max_wait)):
        if cancel_above is not None and hi[t] > cancel_above and not (kind == "limit" and direction == "SELL"):
            return None
        if cancel_below is not None and lo[t] < cancel_below and not (kind == "limit" and direction == "BUY"):
            return None
        if direction == "BUY":
            if kind == "stop" and hi[t] >= level:
                return t, max(level, op[t])
            if kind == "limit" and lo[t] <= level:
                return t, min(level, op[t])
        else:
            if kind == "stop" and lo[t] <= level:
                return t, min(level, op[t])
            if kind == "limit" and hi[t] >= level:
                return t, max(level, op[t])
    return None


def between(x, lo_, hi_, tol=0.05):
    return lo_ * (1 - tol) <= x <= hi_ * (1 + tol)


HARMONICS = {
    # name: (AB/XA range, BC/AB range, CD/BC range, AD/XA target, stop as XA multiple from A)
    "Gartley":   ((0.618, 0.618), (0.382, 0.886), (1.13, 1.618), 0.786, 1.0),
    "Bat":       ((0.382, 0.5),   (0.382, 0.886), (1.618, 2.618), 0.886, 1.0),
    "Butterfly": ((0.786, 0.786), (0.382, 0.886), (1.618, 2.24), 1.27, 1.414),
    "Crab":      ((0.382, 0.618), (0.382, 0.886), (2.24, 3.618), 1.618, 2.0),
}


def find_setups(df, piv):
    """يرجع قائمة صفقات: dict(setup, dir, j(entry idx), entry, stop, target)."""
    out = []
    n = len(piv)
    for m in range(n):
        ci = piv[m][3]  # وقت تأكيد آخر نقطة
        P = piv[max(0, m - 5): m + 1]
        # ---------- إليوت W3 ----------
        if len(P) >= 3:
            (i0, p0, t0, _), (i1, p1, t1, _), (i2, p2, t2, _) = P[-3:]
            w1 = abs(p1 - p0)
            if w1 > 0:
                retr = abs(p1 - p2) / w1
                if t2 == "L" and p2 > p0 and 0.382 <= retr <= 0.786:
                    f = wait_fill(df, ci + 1, "BUY", "stop", p1, cancel_below=p2,
                                  max_wait=next_confirm(piv, m) - ci if m + 1 < n else 40)
                    if f:
                        out.append(dict(setup="Elliott W3", dir="BUY", j=f[0], entry=f[1], stop=p2, target=p2 + 1.618 * w1))
                if t2 == "H" and p2 < p0 and 0.382 <= retr <= 0.786:
                    f = wait_fill(df, ci + 1, "SELL", "stop", p1, cancel_above=p2,
                                  max_wait=next_confirm(piv, m) - ci if m + 1 < n else 40)
                    if f:
                        out.append(dict(setup="Elliott W3", dir="SELL", j=f[0], entry=f[1], stop=p2, target=p2 - 1.618 * w1))
        # ---------- إليوت W5 ----------
        if len(P) >= 5:
            (_, p0, _, _), (_, p1, _, _), (_, p2, _, _), (_, p3, _, _), (_, p4, t4, _) = P[-5:]
            up = t4 == "L"
            s = 1 if up else -1
            w1, w3, w4 = s * (p1 - p0), s * (p3 - p2), s * (p3 - p4)
            if w1 > 0 and w3 > 0 and s * (p2 - p0) > 0 and s * (p3 - p1) > 0 and s * (p4 - p1) > 0 \
                    and w3 >= w1 and 0.236 <= w4 / w3 <= 0.618:
                entry = df["close"].values[ci]
                out.append(dict(setup="Elliott W5", dir="BUY" if up else "SELL", j=ci + 1, entry=entry,
                                stop=p4, target=p4 + s * w1, market=True))
        # ---------- إليوت انعكاس بعد 5 ----------
        if len(P) >= 6:
            (_, p0, _, _), (_, p1, _, _), (_, p2, _, _), (_, p3, _, _), (_, p4, _, _), (_, p5, t5, _) = P[-6:]
            up = t5 == "H"
            s = 1 if up else -1
            w1, w3, w5 = s * (p1 - p0), s * (p3 - p2), s * (p5 - p4)
            if min(w1, w3, w5) > 0 and s * (p2 - p0) > 0 and s * (p4 - p1) > 0 and w3 > min(w1, w5) \
                    and s * (p5 - p3) > 0:
                entry = df["close"].values[ci]
                out.append(dict(setup="Elliott REV after 5", dir="SELL" if up else "BUY", j=ci + 1, entry=entry,
                                stop=p5, target=p5 - s * 0.382 * s * (p5 - p0), market=True))
        # ---------- هارمونك ----------
        if len(P) >= 4:
            (_, X, _, _), (_, A, _, _), (_, B, _, _), (_, C, tC, _) = P[-4:]
            bull = tC == "H"   # X قاع، A قمة، B قاع، C قمة → D قاع (شراء)
            XA, AB, BC = abs(A - X), abs(A - B), abs(C - B)
            if XA > 0 and AB > 0 and BC > 0:
                ab, bc = AB / XA, BC / AB
                for name, (abr, bcr, cdr, adr, stopx) in HARMONICS.items():
                    if not (between(ab, *abr) and between(bc, *bcr)):
                        continue
                    D = A - adr * XA if bull else A + adr * XA
                    cd = abs(C - D) / BC
                    if not between(cd, *cdr, tol=0.1):
                        continue
                    stop = A - stopx * XA * 1.0 - 0.0 if bull else A + stopx * XA
                    if stopx == 1.0:
                        stop = X - 0.05 * XA if bull else X + 0.05 * XA
                    else:
                        stop = A - (stopx + 0.1) * XA if bull else A + (stopx + 0.1) * XA
                    f = wait_fill(df, ci + 1, "BUY" if bull else "SELL", "limit", D,
                                  cancel_above=C if bull else None, cancel_below=None if bull else C,
                                  max_wait=next_confirm(piv, m) - ci if m + 1 < n else 40)
                    if f:
                        AD = abs(A - f[1])
                        for tp in (0.382, 0.618):
                            tgt = f[1] + tp * AD if bull else f[1] - tp * AD
                            out.append(dict(setup=f"Harmonic {name} TP{tp}", dir="BUY" if bull else "SELL",
                                            j=f[0], entry=f[1], stop=stop, target=tgt))
                # Cypher: C يتجاوز A بين 1.13-1.414 من XA، و D = 0.786 من XC
                xc_ext = abs(C - X) / XA
                if between(ab, 0.382, 0.618) and 1.13 <= xc_ext <= 1.414 and ((C > A) if bull else (C < A)):
                    XC = abs(C - X)
                    D = C - 0.786 * XC if bull else C + 0.786 * XC
                    stop = X - 0.05 * XA if bull else X + 0.05 * XA
                    f = wait_fill(df, ci + 1, "BUY" if bull else "SELL", "limit", D,
                                  cancel_above=C if bull else None, cancel_below=None if bull else C,
                                  max_wait=next_confirm(piv, m) - ci if m + 1 < n else 40)
                    if f:
                        CD = abs(C - f[1])
                        for tp in (0.382, 0.618):
                            tgt = f[1] + tp * CD if bull else f[1] - tp * CD
                            out.append(dict(setup=f"Harmonic Cypher TP{tp}", dir="BUY" if bull else "SELL",
                                            j=f[0], entry=f[1], stop=stop, target=tgt))
    return out


def next_confirm(piv, m):
    return piv[m + 1][3] if m + 1 < len(piv) else 10**9


def evaluate(df, setups, split_idx):
    rows = []
    cl = df["close"].values
    for s in setups:
        if s["j"] >= len(df):
            continue
        if (s["dir"] == "BUY" and not s["stop"] < s["entry"] < s["target"]) or \
           (s["dir"] == "SELL" and not s["target"] < s["entry"] < s["stop"]):
            continue
        r = run_trade(df, s["j"], s["dir"], s["entry"], s["stop"], s["target"], not s.get("market"))
        if r is None:
            continue
        # تحكم عشوائي: نفس الاتجاه ونفس المسافات النسبية، دخول بسعر الإغلاق بوقت عشوائي
        sd, td = abs(s["entry"] - s["stop"]) / s["entry"], abs(s["target"] - s["entry"]) / s["entry"]
        ctrl = []
        for _ in range(RANDOM_SAMPLES):
            k = random.randrange(30, len(df) - MAX_HOLD - 1)
            e = cl[k]
            st = e * (1 - sd) if s["dir"] == "BUY" else e * (1 + sd)
            tg = e * (1 + td) if s["dir"] == "BUY" else e * (1 - td)
            rr = run_trade(df, k + 1, s["dir"], e, st, tg, False)
            if rr is not None:
                ctrl.append(rr)
        rows.append(dict(setup=s["setup"], dir=s["dir"], R=r, ctrl=np.mean(ctrl) if ctrl else np.nan,
                         part="train" if s["j"] < split_idx else "test"))
    return rows


def summarize(rows):
    if not rows:
        return None
    r = np.array([x["R"] for x in rows])
    c = np.array([x["ctrl"] for x in rows])
    pf = r[r > 0].sum() / max(1e-9, -r[r < 0].sum())
    return len(r), (r > 0).mean() * 100, r.sum(), r.mean(), pf, np.nanmean(c)


def line(title, rows):
    s = summarize(rows)
    if s is None:
        return f"  {title:34s} لا صفقات"
    n, wr, tot, avg, pf, ctrl = s
    edge = avg - ctrl
    mark = "✅" if (avg > 0 and edge > 0.05) else ("➖" if avg > 0 else "❌")
    return (f"  {mark} {title:32s} صفقات={n:4d} نجاح={wr:5.1f}% صافي={tot:+7.1f}R متوسط={avg:+.3f}R "
            f"PF={pf:4.2f} | العشوائي={ctrl:+.3f}R الميزة={edge:+.3f}R")


print("تحميل البيانات...")
datasets = {}
for c in COINS:
    try:
        d1 = fetch(c, "1d", "max")
        h1 = fetch(c, "1h", "730d")
        if d1 is not None and len(d1) > 400:
            datasets[(c, "1d")] = d1[d1.index >= "2018-01-01"]
        if h1 is not None and len(h1) > 2000:
            datasets[(c, "4h")] = to_4h(h1)
        print(f"  {c}: يومي={0 if d1 is None else len(d1)} | 4س={0 if h1 is None else len(to_4h(h1))}")
    except Exception as e:
        print(f"  {c}: فشل {e}")

ALL = defaultdict(list)
for (coin, tf), df in datasets.items():
    split_idx = int(len(df) * 2 / 3)
    for k in (2.0, 3.0):
        piv = zigzag(df, k)
        rows = evaluate(df, find_setups(df, piv), split_idx)
        for r in rows:
            r.update(coin=coin, tf=tf, k=k)
            ALL[(tf, k)].append(r)

for (tf, k), rows in sorted(ALL.items()):
    for part, title in (("train", "التدريب (أول الثلثين)"), ("test", "الاختبار (الثلث الأخير)")):
        print(f"\n######## فريم {tf} | حساسية ZigZag = {k}×ATR | {title} ########")
        sub = [r for r in rows if r["part"] == part]
        for setup in sorted({r["setup"] for r in sub}):
            print(line(setup, [r for r in sub if r["setup"] == setup]))
            for d in ("BUY", "SELL"):
                rr = [r for r in sub if r["setup"] == setup and r["dir"] == d]
                if rr:
                    print("     " + line(f"↳ {d}", rr).strip())
        print(line("كل إليوت", [r for r in sub if r["setup"].startswith("Elliott")]))
        print(line("كل الهارمونك", [r for r in sub if r["setup"].startswith("Harmonic")]))

print("\n######## الأداء حسب العملة (كل الفترات، أحسن إعداد لكل عائلة) ########")
for fam in ("Elliott", "Harmonic"):
    print(f"\n-- {fam} --")
    by = defaultdict(list)
    for rows in ALL.values():
        for r in rows:
            if r["setup"].startswith(fam):
                by[r["coin"]].append(r)
    for coin, rows in sorted(by.items()):
        print(line(coin, rows))
