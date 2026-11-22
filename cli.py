"""python cli.py [--demo] [--cutoff 10:20] [--k 150]"""
import argparse
import engine as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--cutoff", default="10:20")
    ap.add_argument("--k", type=int, default=150)
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
    s = E.predict(hist, today, cut, a.k)
    V = s["vol"]
    print(f"\nNIFTY {s['ref_price']:.2f} at {a.cutoff}  | library {s['n_hist_days']} days | {V['n']} nearest days used")
    print(f"Morning range {V['morning_range']:.2f}%  ({V['bucket']}, {V['pct_rank']:.0f}th percentile)")
    print(f"Expected afternoon range {V['range_med']:.2f}% (~{V['range_pts']:.0f} pts); middle half {V['range_p25']:.2f}-{V['range_p75']:.2f}%")
    print(f"Bigger than usual ({V['lib_median_range']:.2f}%): {V['p_big']:.0f}% | above 1%: {V['p_gt1']:.0f}% | under 0.5%: {V['p_lt05']:.0f}%")
    print(f"Typical high {V['up_med']:.0f} (stretch {V['up_p75']:.0f}) | typical low {V['dn_med']:.0f} (stretch {V['dn_p25']:.0f})  -- no direction implied")


if __name__ == "__main__":
    main()
