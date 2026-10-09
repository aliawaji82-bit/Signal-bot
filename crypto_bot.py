"""
بوت الكريبتو اليومي - إشارات تجريبية (بدون تنفيذ)
=================================================
مبني على نتائج الاختبار التاريخي (2018 → 2026، 16 عملة، فريم يومي):

1) Donchian20 (الاستراتيجية الأساسية)
   - الشرط: BTC فوق متوسط 200 يوم
   - الدخول: إغلاق العملة فوق أعلى قمة لآخر 20 يوم → شراء مع افتتاح اليوم التالي
   - الوقف الأولي: 2 × ATR(14) تحت الدخول
   - الخروج: إغلاق تحت أدنى قاع لآخر 10 أيام

2) إليوت - دخول الموجة الثالثة (إشارة إضافية)
   - موجة 1 صاعدة ثم تصحيح 38.2%-78.6% (موجة 2) ما يكسر بداية 1، بعد تأكيد قاع 2 بـ ZigZag (3 × ATR)
   - أمر شراء معلّق عند قمة موجة 1، يُلغى لو السعر كسر قاع 2 أو تكوّنت قمة جديدة قبل التنفيذ
   - الوقف: قاع موجة 2 | الهدف: قاع 2 + 2.618 × طول موجة 1

إدارة المخاطرة: 1% من رأس المال لكل صفقة، وبحد أقصى 5 صفقات مفتوحة.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC", "BCH", "TRX", "XLM", "ATOM", "ETC"]

ACCOUNT_BALANCE = float(os.environ.get("ACCOUNT_BALANCE", "1000"))
RISK_PER_TRADE = 0.01
MAX_OPEN = 5

DONCHIAN_IN, DONCHIAN_OUT, DONCHIAN_STOP_ATR = 20, 10, 2.0
ZIGZAG_K = 3.0
W3_RETRACE = (0.382, 0.786)
W3_TARGET_MULT = 2.618
W3_MAX_WAIT_DAYS = 60
W3_MAX_HOLD_DAYS = 250
SUMMARY_EVERY_DAYS = 7

STATE_FILE = "crypto_state.json"
TRADES_FILE = "crypto_trades.json"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PAPER_NOTE = "🧪 إشارة تجريبية - بدون تنفيذ فعلي"


# ============================ أدوات ============================
def send(text):
    print(text + "\n" + "-" * 40)
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)
    except Exception as e:
        print("[خطأ] فشل إرسال تيليجرام:", e)


def fmt(p):
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:.4f}"
    return f"{p:.6f}"


def load(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)


def fetch(coin):
    """يرجع (شموع يومية مقفلة، سعر افتتاح اليوم الحالي أو None)."""
    df = yf.Ticker(f"{coin}-USD").history(period="3y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return None, None
    df.index = pd.to_datetime(df.index, utc=True).normalize()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})[["open", "high", "low", "close"]]
    df = df.astype(float).dropna()
    df = df[~df.index.duplicated(keep="last")]
    today = pd.Timestamp.now(tz="UTC").normalize()
    today_open = None
    if len(df) and df.index[-1] >= today:   # شمعة اليوم لسا ما قفلت
        today_open = float(df["open"].iloc[-1])
        df = df.iloc[:-1]
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, min_periods=14).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    return df, today_open


def zigzag(df, k):
    """نفس خوارزمية الاختبار: (idx, price, 'H'/'L', confirm_idx)."""
    hi, lo, a = df["high"].values, df["low"].values, df["atr"].values
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


def day(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def size_text(entry, stop):
    risk_usd = ACCOUNT_BALANCE * RISK_PER_TRADE
    units = risk_usd / (entry - stop)
    return (f"الحجم المقترح: {units:.6g} وحدة ≈ ${units * entry:,.0f} "
            f"(مخاطرة {RISK_PER_TRADE * 100:.0f}% = ${risk_usd:.0f} من ${ACCOUNT_BALANCE:,.0f})")


# ============================ تتبع الصفقات المفتوحة ============================
def update_open_trade(t, df):
    """يمشي على الشموع المقفلة من بعد آخر فحص. يرجع (سعر الخروج، التاريخ، السبب) أو None."""
    entry_day = pd.Timestamp(t["entry_date"], tz="UTC")
    last = pd.Timestamp(t.get("checked", t["entry_date"]), tz="UTC")
    lo, hi, cl, op = df["low"], df["high"], df["close"], df["open"]
    for i in range(len(df)):
        ts = df.index[i]
        if ts < entry_day or (ts <= last and t.get("checked")):
            continue
        is_entry_bar = ts == entry_day
        if lo.iloc[i] <= t["stop"]:
            px = t["stop"] if is_entry_bar else min(t["stop"], op.iloc[i])
            return px, ts, "ضرب الوقف"
        if t["strategy"] == "Donchian20":
            if i >= DONCHIAN_OUT and cl.iloc[i] < lo.iloc[i - DONCHIAN_OUT:i].min():
                return cl.iloc[i], ts, f"إغلاق تحت أدنى قاع {DONCHIAN_OUT} أيام"
        else:
            if not is_entry_bar and hi.iloc[i] >= t["target"]:
                return t["target"], ts, "وصل الهدف"
            if (ts - entry_day).days >= W3_MAX_HOLD_DAYS - 1:
                return cl.iloc[i], ts, "انتهت المدة القصوى"
        t["checked"] = day(ts)
    return None


def close_msg(t):
    pct = (t["exit_price"] / t["entry"] - 1) * 100
    icon = "✅" if t["R"] > 0 else "❌"
    return (f"{icon} إغلاق صفقة - {t['coin']} ({t['strategy_ar']})\n"
            f"الدخول: {fmt(t['entry'])} بتاريخ {t['entry_date']}\n"
            f"الخروج: {fmt(t['exit_price'])} بتاريخ {t['exit_date']} ({t['exit_reason']})\n"
            f"النتيجة: {pct:+.1f}% من سعر الدخول = {t['R']:+.2f}R\n{PAPER_NOTE}")


def finish(t, px, ts, reason):
    t.update(status="closed", exit_price=float(px), exit_date=day(ts), exit_reason=reason,
             R=round((float(px) - t["entry"]) / (t["entry"] - t["initial_stop"]), 3))


# ============================ الرئيسي ============================
def main():
    state = load(STATE_FILE, {"pending": [], "seen_setups": [], "last_summary": None})
    trades = load(TRADES_FILE, [])
    data = {}
    for c in COINS:
        try:
            df, today_open = fetch(c)
            if df is not None and len(df) > 210:
                data[c] = (df, today_open)
        except Exception as e:
            print(f"[تحذير] فشل جلب {c}: {e}")
    if "BTC" not in data:
        print("ما قدرنا نجيب بيانات BTC - نوقف هالمرة")
        return
    btc = data["BTC"][0]
    btc_up = bool(btc["close"].iloc[-1] > btc["sma200"].iloc[-1])
    last_day = btc.index[-1]
    print(f"آخر شمعة مقفلة: {day(last_day)} | BTC فوق متوسط 200: {btc_up}")

    # ---------- 1) تحديث الصفقات المفتوحة ----------
    for t in [x for x in trades if x["status"] == "open"]:
        if t["coin"] not in data:
            continue
        res = update_open_trade(t, data[t["coin"]][0])
        if res:
            finish(t, *res)
            send(close_msg(t))

    def open_count():
        return sum(1 for x in trades if x["status"] == "open")

    # ---------- 2) الأوامر المعلّقة لإليوت ----------
    still = []
    for p in state["pending"]:
        if p["coin"] not in data:
            still.append(p)
            continue
        df = data[p["coin"]][0]
        piv = zigzag(df, ZIGZAG_K)
        later = [df.index[c] for (_, _, _, c) in piv if df.index[c] > pd.Timestamp(p["confirm_date"], tz="UTC")]
        next_pivot = min(later) if later else None
        outcome = None
        for i in range(len(df)):
            ts = df.index[i]
            if ts <= pd.Timestamp(p.get("checked", p["confirm_date"]), tz="UTC"):
                continue
            if next_pivot is not None and ts > next_pivot:
                outcome = ("cancel", ts, "تكوّنت قمة جديدة تحت قمة الموجة 1")
                break
            if df["low"].iloc[i] < p["p2"]:
                outcome = ("cancel", ts, "السعر كسر قاع الموجة 2")
                break
            if df["high"].iloc[i] >= p["p1"]:
                outcome = ("fill", ts, max(p["p1"], df["open"].iloc[i]))
                break
            p["checked"] = day(ts)
        if outcome is None:
            age = (last_day - pd.Timestamp(p["confirm_date"], tz="UTC")).days
            if age >= W3_MAX_WAIT_DAYS:
                outcome = ("cancel", last_day, "انتهت مدة الانتظار")
        if outcome is None:
            still.append(p)
            continue
        if outcome[0] == "cancel":
            send(f"🚫 إلغاء أمر معلّق - {p['coin']} (إليوت موجة 3)\nالسبب: {outcome[2]}\n"
                 f"إذا حاط الأمر بمنصتك، ألغه.\n{PAPER_NOTE}")
            continue
        if open_count() >= MAX_OPEN:
            send(f"⚠️ أمر {p['coin']} (إليوت موجة 3) تنفّذ بس ما انحسب - عندك {MAX_OPEN} صفقات مفتوحة (الحد الأقصى)\n{PAPER_NOTE}")
            continue
        entry = float(outcome[2])
        t = dict(id=f"{p['coin']}-W3-{day(outcome[1])}", coin=p["coin"], strategy="ElliottW3", strategy_ar="إليوت موجة 3",
                 entry=entry, entry_date=day(outcome[1]), stop=p["p2"], initial_stop=p["p2"], target=p["target"], status="open")
        trades.append(t)
        res = update_open_trade(t, df)   # لو ضرب الوقف بنفس الشمعة أو بعدها
        send(f"🟢 تنفّذ أمر الشراء - {p['coin']} (إليوت موجة 3)\nسعر الدخول: {fmt(entry)} بتاريخ {t['entry_date']}\n"
             f"الوقف: {fmt(t['stop'])} | الهدف: {fmt(t['target'])}\n{PAPER_NOTE}")
        if res:
            finish(t, *res)
            send(close_msg(t))
    state["pending"] = still

    # ---------- 3) إعدادات جديدة لإليوت (أمر معلّق) ----------
    seen = set(state["seen_setups"])
    for c, (df, _) in data.items():
        piv = zigzag(df, ZIGZAG_K)
        for m in range(2, len(piv)):
            (i0, p0, _, _), (i1, p1, _, _), (i2, p2, t2, c2) = piv[m - 2: m + 1]
            key = f"{c}-{day(df.index[i2])}"
            if key in seen or t2 != "L" or p2 <= p0:
                continue
            if (last_day - df.index[c2]).days > W3_MAX_WAIT_DAYS:
                continue
            w1 = p1 - p0
            if w1 <= 0 or not (W3_RETRACE[0] <= (p1 - p2) / w1 <= W3_RETRACE[1]):
                continue
            seen.add(key)
            # هل لسا معلّق اليوم؟ (ما نلحق إعداد تنفّذ أو انلغى قبل ما نشوفه)
            after = df.iloc[c2 + 1:]
            if (after["low"] < p2).any() or (after["high"] >= p1).any() or m + 1 < len(piv):
                continue
            target = p2 + W3_TARGET_MULT * w1
            p = dict(coin=c, p0=float(p0), p1=float(p1), p2=float(p2), target=float(target),
                     p2_date=day(df.index[i2]), confirm_date=day(df.index[c2]))
            state["pending"].append(p)
            send(f"⏳ أمر شراء معلّق جديد - {c} (إليوت: دخول الموجة 3)\n"
                 f"ضع أمر شراء (Buy Stop) عند: {fmt(p1)} (قمة الموجة 1)\n"
                 f"الوقف: {fmt(p2)} (قاع الموجة 2) | الهدف: {fmt(target)}\n"
                 f"{size_text(p1, p2)}\n"
                 f"يُلغى الأمر لو نزل السعر تحت {fmt(p2)} قبل التنفيذ\n{PAPER_NOTE}")
    state["seen_setups"] = sorted(seen)[-2000:]

    # ---------- 4) إشارات Donchian20 الجديدة ----------
    if not btc_up:
        print("BTC تحت متوسط 200 يوم → ما فيه إشارات Donchian جديدة")
    else:
        for c, (df, today_open) in data.items():
            if any(x["status"] == "open" and x["coin"] == c and x["strategy"] == "Donchian20" for x in trades):
                continue
            if df["close"].iloc[-1] <= df["high"].iloc[-DONCHIAN_IN - 1:-1].max():
                continue
            if open_count() >= MAX_OPEN:
                send(f"⚠️ إشارة شراء {c} (اختراق 20 يوم) بس عندك {MAX_OPEN} صفقات مفتوحة - تجاهلناها\n{PAPER_NOTE}")
                continue
            entry = today_open if today_open else float(df["close"].iloc[-1])
            stop = entry - DONCHIAN_STOP_ATR * float(df["atr"].iloc[-1])
            today = (last_day + pd.Timedelta(days=1))
            exit_level = float(df["low"].iloc[-DONCHIAN_OUT:].min())
            t = dict(id=f"{c}-D20-{day(today)}", coin=c, strategy="Donchian20", strategy_ar="اختراق 20 يوم",
                     entry=float(entry), entry_date=day(today), stop=float(stop), initial_stop=float(stop), status="open")
            trades.append(t)
            send(f"🟢 إشارة شراء - {c} (اختراق أعلى قمة 20 يوم)\n"
                 f"الدخول: الآن بسعر السوق (افتتاح اليوم ≈ {fmt(entry)})\n"
                 f"الوقف الأولي: {fmt(stop)} ({(stop / entry - 1) * 100:.1f}%)\n"
                 f"الخروج: لما يقفل يومي تحت أدنى قاع 10 أيام (حالياً ≈ {fmt(exit_level)}) - البوت ينبهك\n"
                 f"{size_text(entry, stop)}\n{PAPER_NOTE}")

    # ---------- 5) ملخص أسبوعي ----------
    now = datetime.now(timezone.utc)
    last_sum = state.get("last_summary")
    if not last_sum or (now - datetime.fromisoformat(last_sum)).days >= SUMMARY_EVERY_DAYS:
        closed = [x for x in trades if x["status"] == "closed"]
        opened = [x for x in trades if x["status"] == "open"]
        lines = ["📊 ملخص بوت الكريبتو (تجريبي)",
                 f"BTC {'فوق' if btc_up else 'تحت'} متوسط 200 يوم {'✅ إشارات الاختراق شغّالة' if btc_up else '⛔ إشارات الاختراق متوقفة'}"]
        if closed:
            r = np.array([x["R"] for x in closed])
            eq = 1.0
            for x in sorted(closed, key=lambda x: x["exit_date"]):
                eq *= 1 + RISK_PER_TRADE * x["R"]
            lines += [f"صفقات مقفلة: {len(r)} | رابحة: {(r > 0).sum()} | نسبة النجاح: {(r > 0).mean() * 100:.0f}%",
                      f"صافي: {r.sum():+.1f}R ≈ {(eq - 1) * 100:+.1f}% على الحساب (بمخاطرة 1%)"]
            for s, ar in (("Donchian20", "اختراق 20 يوم"), ("ElliottW3", "إليوت موجة 3")):
                rs = [x["R"] for x in closed if x["strategy"] == s]
                if rs:
                    lines.append(f"  {ar}: {len(rs)} صفقة، {np.sum(rs):+.1f}R")
        else:
            lines.append("لسا ما فيه صفقات مقفلة")
        lines.append(f"صفقات مفتوحة: {len(opened)}" + (": " + ", ".join(f"{x['coin']}" for x in opened) if opened else ""))
        lines.append(f"أوامر معلّقة: {len(state['pending'])}" + (": " + ", ".join(p["coin"] for p in state["pending"]) if state["pending"] else ""))
        lines.append("المتوقع من الاختبار التاريخي: نسبة نجاح ~35-40% والربح من صفقات قليلة كبيرة - الحكم يحتاج شهور")
        send("\n".join(lines))
        state["last_summary"] = now.isoformat()

    save(STATE_FILE, state)
    save(TRADES_FILE, trades)


if __name__ == "__main__":
    main()
