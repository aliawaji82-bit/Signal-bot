"""فحص متانة لأفضل المرشحين من crypto_research.py: الوسيط، أثر الصفقات الشاذة، كل سنة، كل عملة،
ومحاكاة حساب بحد أقصى 5 صفقات متزامنة."""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "crypto_research.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(here, "crypto_research.py")}
exec(src.split("# ---------------- الشبكة ----------------")[0], ns)
run_config, control, D, SPLIT = ns["run_config"], ns["control"], ns["D"], ns["SPLIT"]

CANDS = {
    "Donchian20 دخول، خروج كسر قاع 10 أيام، BTC>200": ns["donchian_gen"](20, "donchian10", "btc>200"),
    "Donchian20 دخول، خروج كسر قاع 10 أيام، العملة>200": ns["donchian_gen"](20, "donchian10", "coin>200"),
    "Donchian20 دخول، خروج كسر قاع 20 يوم، BTC>200": ns["donchian_gen"](20, "donchian20", "btc>200"),
    "Donchian55 دخول، خروج كسر قاع 10 أيام، BTC>200": ns["donchian_gen"](55, "donchian10", "btc>200"),
    "Donchian55 دخول، خروج كسر قاع 20 يوم، BTC>200": ns["donchian_gen"](55, "donchian20", "btc>200"),
    "Donchian20 بدون فلتر، خروج 10": ns["donchian_gen"](20, "donchian10", "none"),
    "إليوت W3 k3 هدف 2.618 بدون فلتر": ns["w3_gen"](3.0, (0.382, 0.786), "w2", "tp2.618", "none"),
    "إليوت W3 k3 هدف 1.618 بدون فلتر": ns["w3_gen"](3.0, (0.382, 0.786), "w2", "tp1.618", "none"),
    "إليوت W3 k2.5 وقف متحرك 4ATR، BTC والعملة>200": ns["w3_gen"](2.5, (0.382, 0.786), "w2", "trail4.0", "btc&coin>200"),
    "إليوت W3 k2.5 وقف متحرك 4ATR، BTC>200": ns["w3_gen"](2.5, (0.382, 0.786), "w2", "trail4.0", "btc>200"),
}


def robust(rows, title):
    if not rows:
        print(f"  {title}: لا صفقات"); return
    r = np.sort(np.array([x["R"] for x in rows]))
    c = np.array([x["ctrl"] for x in rows])
    cut = max(1, int(len(r) * 0.02))
    top10 = r[-10:].sum()
    print(f"  {title:8s} صفقات={len(r):4d} نجاح={(r > 0).mean() * 100:4.1f}% متوسط={r.mean():+.2f}R وسيط={np.median(r):+.2f}R "
          f"متوسط بدون أعلى 2%={r[:-cut].mean():+.2f}R | عشوائي={np.nanmean(c):+.2f}R (بدون أعلى 2%: "
          f"{np.nanmean(np.sort(c)[:-cut]):+.2f}R) | حصة أعلى 10 صفقات من الربح={top10 / max(1e-9, r.sum()) * 100:5.0f}%")


def simulate(rows, start, end=None, risk=0.01, max_open=5):
    rows = sorted([x for x in rows if x["t_in"] >= start and (end is None or x["t_in"] < end)], key=lambda x: x["t_in"])
    ev = sorted([(x["t_in"], 1, i) for i, x in enumerate(rows)] + [(x["t_out"], 0, i) for i, x in enumerate(rows)],
                key=lambda e: (e[0], e[1]))
    eq, peak, mdd, open_risk, taken = 1.0, 1.0, 0.0, {}, 0
    for t, typ, i in ev:
        if typ == 1:
            if len(open_risk) < max_open:
                open_risk[i] = eq * risk; taken += 1
        elif i in open_risk:
            eq += rows[i]["R"] * open_risk.pop(i)
            peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    return eq - 1, mdd, taken


btc = D["BTC"]["close"]
for name, gen in CANDS.items():
    rows = run_config(name, gen)
    for x in rows:
        x["ctrl"] = control(D[x["coin"]], x["tr"], n=15)
    print(f"\n==================== {name} ====================")
    robust([x for x in rows if x["part"] == "train"], "تدريب")
    robust([x for x in rows if x["part"] == "test"], "اختبار")
    by_year = defaultdict(list)
    for x in rows:
        by_year[x["t_in"].year].append(x["R"])
    print("  حسب السنة: " + " | ".join(f"{y}: {len(v)} صفقة {np.sum(v):+.0f}R" for y, v in sorted(by_year.items())))
    by_coin = defaultdict(list)
    for x in rows:
        by_coin[x["coin"]].append(x["R"])
    pos = sum(1 for v in by_coin.values() if np.sum(v) > 0)
    print(f"  عملات رابحة: {pos} من {len(by_coin)} | " + " ".join(f"{c}:{np.sum(v):+.0f}" for c, v in sorted(by_coin.items())))
    for lbl, s, e in (("تدريب 2018-2023", pd.Timestamp("2018-01-01", tz="UTC"), SPLIT), ("اختبار 2023-2026", SPLIT, None),
                      ("سنة 2022 الهابطة", pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"))):
        ret, mdd, n = simulate(rows, s, e)
        bs = btc[btc.index >= s]; be = btc[btc.index < e] if e is not None else btc
        bh = be.iloc[-1] / bs.iloc[0] - 1
        print(f"  حساب (1% مخاطرة، أقصى 5 صفقات) {lbl}: العائد={ret * 100:+7.1f}% أقصى تراجع={mdd * 100:5.1f}% صفقات={n} | BTC شراء واحتفاظ={bh * 100:+.0f}%")
