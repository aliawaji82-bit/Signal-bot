"""
إعادة تقييم كل الإشارات المرسلة من أول يوم ببيانات أسعار حقيقية (شموع 5 دقائق من Yahoo).
لكل إشارة: نمشي على الشموع بعد وقت الإرسال بالترتيب ونشوف أيهم انلمس أول: الهدف أو الوقف.
لو الاثنين بنفس الشمعة ما نقدر نعرف مين أول (نحسبها "غير محسومة" ونعرضها لحالها).
"""
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd
import yfinance as yf

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "x")
os.environ.setdefault("TELEGRAM_CHAT_ID", "x")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scanner  # noqa: E402

YF_OF = {c["label"]: c["yf"] for c in scanner.SYMBOLS}
CAT_OF = {c["label"]: c["category"] for c in scanner.SYMBOLS}
RR = scanner.TP_ATR_MULT / scanner.SL_ATR_MULT

log = json.load(open("signals_log.json"))
data = {}
for label, sym in YF_OF.items():
    try:
        df = yf.Ticker(sym).history(period="60d", interval="5m", auto_adjust=False)
        df.index = pd.to_datetime(df.index, utc=True)
        data[label] = df[["High", "Low", "Close"]]
        print(f"{label}: {len(df)} شمعة من {df.index.min()} إلى {df.index.max()}")
    except Exception as e:
        print(f"{label}: فشل ({e})")

results = []
for s in log:
    df = data.get(s["symbol"])
    opened = pd.Timestamp(s["opened_at"])
    res = {"symbol": s["symbol"], "cat": CAT_OF.get(s["symbol"]), "dir": s["direction"],
           "opened": s["opened_at"], "old": s["status"]}
    if df is None or df.empty or opened < df.index.min():
        res["new"] = "no_data"
        results.append(res)
        continue
    bars = df[df.index >= opened]
    outcome, mins = "open", None
    for t, b in bars.iterrows():
        if s["direction"] == "BUY":
            hit_sl, hit_tp = b["Low"] <= s["sl"], b["High"] >= s["tp"]
        else:
            hit_sl, hit_tp = b["High"] >= s["sl"], b["Low"] <= s["tp"]
        if hit_sl and hit_tp:
            outcome = "ambiguous"
        elif hit_tp:
            outcome = "win"
        elif hit_sl:
            outcome = "loss"
        else:
            continue
        mins = (t - opened).total_seconds() / 60
        break
    res["new"], res["minutes"] = outcome, mins
    results.append(res)


def summarize(rows, title):
    c = Counter(r["new"] for r in rows)
    w, l, a = c["win"], c["loss"], c["ambiguous"]
    decided = w + l
    wr = 100 * w / decided if decided else 0
    net = w * RR - l
    print(f"{title}: إشارات={len(rows)} | ربح={w} | خسارة={l} | غير محسومة={a} | مفتوحة={c['open']} | بدون بيانات={c['no_data']}"
          f" | نسبة النجاح={wr:.1f}% | صافي={net:+.0f}R")
    return {"n": len(rows), "win": w, "loss": l, "ambiguous": a, "open": c["open"], "no_data": c["no_data"],
            "win_rate": round(wr, 1), "net_R": round(net, 1)}


print("\n" + "=" * 80)
print(f"نسبة الهدف للوقف = {RR:.1f} → نقطة التعادل = {100 / (1 + RR):.1f}%")
report = {"overall": summarize(results, "الإجمالي")}
print("\n--- حسب نوع السوق ---")
report["by_cat"] = {k: summarize([r for r in results if r["cat"] == k], k) for k in sorted({r["cat"] for r in results if r["cat"]})}
print("\n--- حسب الاتجاه ---")
report["by_dir"] = {k: summarize([r for r in results if r["dir"] == k], k) for k in ("BUY", "SELL")}
print("\n--- حسب الأسبوع ---")
weeks = defaultdict(list)
for r in results:
    weeks[pd.Timestamp(r["opened"]).to_period("W").start_time.date().isoformat()].append(r)
report["by_week"] = {k: summarize(v, k) for k, v in sorted(weeks.items())}
print("\n--- حسب الرمز ---")
report["by_symbol"] = {k: summarize([r for r in results if r["symbol"] == k], k) for k in sorted({r["symbol"] for r in results})}
print("\n--- مقارنة بالسجل القديم (الطريقة الغلط) ---")
cmp = Counter((r["old"], r["new"]) for r in results)
for k, v in sorted(cmp.items(), key=lambda x: -x[1]):
    print(f"  قديم={k[0]:5s} → فعلي={k[1]:9s}: {v}")
report["old_vs_new"] = {f"{a}->{b}": v for (a, b), v in cmp.items()}
mins = sorted(r["minutes"] for r in results if r.get("minutes") is not None and r["new"] in ("win", "loss"))
if mins:
    print(f"\nمتوسط مدة الصفقة حتى الإقفال: {sum(mins) / len(mins):.0f} دقيقة | الوسيط: {mins[len(mins) // 2]:.0f} دقيقة")
json.dump({"report": report, "signals": results}, open("analysis/results.json", "w"), ensure_ascii=False, indent=1, default=str)
