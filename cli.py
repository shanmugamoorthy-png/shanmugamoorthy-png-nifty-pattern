"""python cli.py [--demo] [--cutoff 10:20] [--k 20] [--weekday]"""
import argparse
import engine as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--cutoff", default="10:20")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--weekday", action="store_true")
    a = ap.parse_args()
    h, m = map(int, a.cutoff.split(":"))
    cut = h * 60 + m
    if a.demo:
        import demo
        hist = demo.make_demo()
        today = demo.make_today(hist, cut)
    else:
        import dhan_history as D
        hist, today = D.update_history(progress=print), D.fetch_today()
    s = E.predict(hist, today, cut, a.k, a.weekday)
    print(f"\nNIFTY {s['ref_price']:.2f} at {a.cutoff}  | library {s['n_hist_days']} days | top {s['k']}")
    print(f"Closed UP {s['up']} / DOWN {s['down']}   median move to close {s['median_ret']:+.2f}%")
    print(f"Expected close {s['exp_close']:.0f}  (25-75%: {s['p25_close']:.0f} - {s['p75_close']:.0f})")
    print(f"Expected high {s['exp_high']:.0f} around {s['median_high_time']}  | low {s['exp_low']:.0f} around {s['median_low_time']}")
    for name, c in s["buckets"]:
        print(f"  {name:30s} {c:2d} days")
    print("Stability (neighbours: up count, median %):", s["by_k"])
    print(s["top"][["date", "weekday", "similarity", "morning_ret", "ret_close", "mfe", "mae"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
