"""Analyze client playback telemetry (logs/telemetry/<session>.jsonl).

Each line is one turn recorded by index.html. We only ever look at *deltas* within a single
clock: client_recv_ts / performance.now() and server_send_ts / perf_counter() live on unsynced
clocks, so their absolute difference is meaningless. server_send_ts is stored for reference but
only its consecutive deltas are analyzed.

    python scripts/analyze_telemetry.py logs/telemetry/20260730_120000.jsonl
    python scripts/analyze_telemetry.py --selftest
"""
import argparse
import json
import math
import sys

FANOUT_RATIO = 1.5   # last-third median inter-arrival > first-third * this => "fanning out"


def pct(xs, p):
    """Linear-interpolated percentile (p in 0..100). None for empty input."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def deltas(xs):
    return [b - a for a, b in zip(xs, xs[1:])]


def fanout(ia):
    """Compare median inter-arrival of the first third vs last third of a turn's chunks."""
    third = len(ia) // 3
    if third < 1:
        return None
    first, last = pct(ia[:third], 50), pct(ia[-third:], 50)
    ratio = (last / first) if first else None
    return {"first_third_med_ms": first, "last_third_med_ms": last,
            "ratio": ratio, "fanned": ratio is not None and ratio > FANOUT_RATIO}


def analyze_turn(t):
    chunks = t.get("chunks", [])
    recv = [c["client_recv_ts"] for c in chunks if c.get("client_recv_ts") is not None]
    ssend = [c["server_send_ts"] for c in chunks if c.get("server_send_ts") is not None]
    ia = deltas(recv)                          # client inter-arrival, ms (performance.now)
    sd = [x * 1000 for x in deltas(ssend)]     # server inter-send, s -> ms (perf_counter)
    return {"turn_id": t.get("turn_id"), "chunk_count": t.get("chunk_count", len(chunks)),
            "underrun_count": t.get("underrun_count"), "total_gap_ms": t.get("total_gap_ms"),
            "ttf_play_ms": t.get("time_to_first_audio_play"),
            "ttf_recv_ms": t.get("time_to_first_audio_recv"),
            "ia": ia, "sd": sd, "fanout": fanout(ia)}


def _fmt(x, unit=""):
    return "n/a" if x is None else f"{x:.1f}{unit}"


def report(records):
    turns = [analyze_turn(t) for t in records]
    print(f"== {len(turns)} turn(s) ==\n")
    for t in turns:
        fo = t["fanout"]
        fo_s = ("n/a" if not fo else
                f"first={_fmt(fo['first_third_med_ms'])}ms last={_fmt(fo['last_third_med_ms'])}ms "
                f"ratio={_fmt(fo['ratio'])} {'FANS OUT' if fo['fanned'] else 'flat'}")
        print(f"turn {t['turn_id']}  chunks={t['chunk_count']}  "
              f"underruns={t['underrun_count']} gap={_fmt(t['total_gap_ms'],'ms')}  "
              f"ttf_play={_fmt(t['ttf_play_ms'],'ms')}")
        print(f"    inter-arrival ms: p50={_fmt(pct(t['ia'],50))} p95={_fmt(pct(t['ia'],95))} "
              f"max={_fmt(max(t['ia']) if t['ia'] else None)}")
        print(f"    fan-out: {fo_s}")

    all_ia = [x for t in turns for x in t["ia"]]
    ttfs = [t["ttf_play_ms"] for t in turns if t["ttf_play_ms"] is not None]
    evaluated = [t for t in turns if t["fanout"]]
    fanned = [t for t in evaluated if t["fanout"]["fanned"]]
    print("\n== aggregate ==")
    print(f"underruns total: {sum(t['underrun_count'] or 0 for t in turns)}   "
          f"total gap: {_fmt(sum(t['total_gap_ms'] or 0 for t in turns),'ms')}")
    print(f"time to first audio play: p50={_fmt(pct(ttfs,50))}ms p95={_fmt(pct(ttfs,95))}ms")
    print(f"inter-arrival ms (all chunks): p50={_fmt(pct(all_ia,50))} "
          f"p95={_fmt(pct(all_ia,95))} max={_fmt(max(all_ia) if all_ia else None)}")
    print(f"fan-out: {len(fanned)}/{len(evaluated)} turns fan out "
          f"(inter-arrival grows first->last third by >{FANOUT_RATIO}x)")


def _selftest():
    # growing gaps -> fans out; constant gaps -> flat.
    grow = {"turn_id": "t01", "chunk_count": 6, "underrun_count": 2, "total_gap_ms": 345.0,
            "time_to_first_audio_play": 250.0,
            "chunks": [{"client_recv_ts": r, "server_send_ts": r / 1000.0}
                       for r in [0, 100, 210, 340, 500, 700]]}
    flat = {"turn_id": "t02", "chunk_count": 6, "underrun_count": 0, "total_gap_ms": 0.0,
            "time_to_first_audio_play": 200.0,
            "chunks": [{"client_recv_ts": r, "server_send_ts": r / 1000.0}
                       for r in [0, 100, 200, 300, 400, 500]]}
    g, f = analyze_turn(grow), analyze_turn(flat)
    assert g["ia"] == [100, 110, 130, 160, 200], g["ia"]
    assert pct(g["ia"], 50) == 130 and max(g["ia"]) == 200
    assert [round(x, 6) for x in g["sd"]] == [100, 110, 130, 160, 200], g["sd"]   # server deltas: s -> ms
    assert g["fanout"]["fanned"] is True, g["fanout"]
    assert f["fanout"]["fanned"] is False, f["fanout"]
    assert pct([250.0, 200.0], 50) == 225.0
    assert fanout([100, 100]) is None                         # <3 deltas: not enough to judge
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="telemetry JSONL file")
    ap.add_argument("--selftest", action="store_true", help="run built-in checks and exit")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.path:
        ap.error("give a JSONL path or --selftest")
    with open(args.path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if not records:
        print("no turns in file", file=sys.stderr)
        return
    report(records)


if __name__ == "__main__":
    main()
