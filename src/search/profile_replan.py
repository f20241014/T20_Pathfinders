"""Profile agent replanning on the real map (R7, Week 6).

Runs R1's simulator with R7's AgentPlanner (independent or full mode) and
reports, from the replan rows each run logs (D9):

* CPU and wall time per replan, by trigger (median, p95, max, in ms);
* how many replans read 0.0 CPU (the timer's tick is coarser than the
  replan; Windows ticks every ~15.6 ms, so use wall_seconds there);
* where the time goes, from cProfile on one run: D* Lite repair, the
  conflict check and auction (R5), ETA building, cost refresh, sync, the
  rest of the simulator and logging.

    python -m src.search.profile_replan                       # 3 seeds, full mode
    python -m src.search.profile_replan --mode independent --seeds 5

Writes reports/replan_profile.csv (one row per replan) unless --no-csv.
This is profiling, not the experiment: the Week 7 analysis uses R8's frozen
configs.
"""

import argparse
import cProfile
import csv
import io
import json
import math
import os
import pstats
import random
import statistics
import tempfile
import time
from collections import defaultdict

from src.coordination.trigger import ThresholdTrigger
from src.search.astar import astar
from src.search.heuristics import make_heuristic
from src.sim.agent_planner import AgentPlanner
from src.sim.coordination_hook import TriggerCoordination
from src.sim.environment import DeliverySimEnvironment
from src.sim.models import Order, Vehicle
from src.sim.world import FakeWorld
from src.world.graph_loader import load_road_graph

CSV_PATH = os.path.join("reports", "replan_profile.csv")

# cProfile functions -> buckets, by file and function name (first match wins)
BUCKETS = [
    ("D* Lite repair", ("dstar_lite.py", None)),
    ("conflict check + auction (R5)", ("src/coordination", None)),
    ("ETA building", ("sim_replanner.py", "to_route")),
    ("cost refresh (D10)", ("sim_replanner.py", "refresh")),
    ("hysteresis compare", ("sim_replanner.py", "path_cost_of")),
    ("multi-stop bookkeeping", ("multi_stop.py", None)),
    ("logging", ("logger", None)),
]


def build_run(G, heuristic, seed, mode, n_vehicles, n_stops, rate, log_dir):
    """One scenario: n_vehicles x n_stops random stops, all ending at one hub."""
    world = FakeWorld(G)
    env = DeliverySimEnvironment(world, {
        "run_id": f"profile_{mode}_{seed}", "seed": seed, "mode": mode,
        "log_path": os.path.join(log_dir, f"{mode}_{seed}.jsonl"),
        "disruption_rate_per_hour": rate,
    })
    hook = TriggerCoordination(ThresholdTrigger(tau=600.0)) if mode == "full" else None
    planner = AgentPlanner(env, mode=mode, heuristic=heuristic, coordination=hook)
    rnd = random.Random(seed)
    nodes = sorted(G.nodes)
    hub = rnd.choice(nodes)
    for i in range(n_vehicles):
        while True:
            pick = rnd.sample(nodes, n_stops)
            start, stops = pick[0], pick[1:] + [hub]
            if all(astar(G, x, y).cost < math.inf for x, y in zip([start] + stops, stops)):
                break
        veh = Vehicle(f"V{i}", start, 10.0, current_node=start)
        orders = [Order(f"o{i}_{j}", n, 1.0, service_time=0.0) for j, n in enumerate(stops)]
        planner.register(veh, orders, stops=stops)
        env.add_vehicle(veh, orders)
    return env, planner


def read_rows(path, events=("replan", "replan_kept")):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["event"] in events:
                out.append(r)
    return out


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bucket_of(filename, func):
    name = filename.replace("\\", "/")
    for label, (where, fn) in BUCKETS:
        if where in name and (fn is None or fn == func):
            return label
    return None


def profile_buckets(stats):
    """Own time (tottime) per bucket, in seconds, plus the total."""
    out = defaultdict(float)
    total = 0.0
    for (filename, _line, func), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():
        total += tt
        label = bucket_of(filename, func)
        if label is None:
            label = "simulator, Python and the rest"
        out[label] += tt
    return dict(out), total


def summarise(rows):
    by = defaultdict(list)
    for r in rows:
        key = r["event"] if r["event"] == "replan_kept" else r["payload"]["trigger"]
        by[key].append(r["payload"])
    lines = []
    for key in sorted(by):
        p = by[key]
        cpu = [x["cpu_seconds"] * 1000 for x in p]
        wall = [x.get("wall_seconds", float("nan")) * 1000 for x in p]
        lines.append({
            "kind": key, "n": len(p),
            "cpu_med_ms": statistics.median(cpu), "cpu_p95_ms": pct(cpu, 0.95),
            "cpu_max_ms": max(cpu), "wall_med_ms": statistics.median(wall),
            "wall_p95_ms": pct(wall, 0.95), "wall_max_ms": max(wall),
            "zero_cpu": sum(1 for x in cpu if x == 0.0),
        })
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", default="secunderabad_small")
    ap.add_argument("--mode", choices=("independent", "full"), default="full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--vehicles", type=int, default=8)
    ap.add_argument("--stops", type=int, default=4)
    ap.add_argument("--rate", type=float, default=600.0, help="disruptions per hour")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args(argv)

    G = load_road_graph(args.graph)
    heuristic = make_heuristic(G)
    log_dir = tempfile.mkdtemp(prefix="replan_profile_")
    rows, prof = [], None
    totals = defaultdict(float)
    for seed in range(args.seeds):
        env, planner = build_run(G, heuristic, seed, args.mode, args.vehicles, args.stops,
                                 args.rate, log_dir)
        t0, c0 = time.perf_counter(), time.process_time()
        if seed == 0:
            prof = cProfile.Profile()
            prof.enable()
        env.run(args.hours * 3600.0)
        if seed == 0:
            prof.disable()
        totals["run_wall_s"] += time.perf_counter() - t0
        totals["run_cpu_s"] += time.process_time() - c0
        s = planner.finish()
        for k in ("replan_cpu_seconds", "kept_cpu_seconds", "agent_cpu_seconds",
                  "coordination_cpu_seconds"):
            totals[k] += s[k]
        for r in read_rows(env.log_path):
            r["seed"] = seed
            rows.append(r)

    info = time.get_clock_info("process_time")
    print(f"graph {args.graph}, mode {args.mode}, {args.seeds} seeds, {args.vehicles} vehicles x "
          f"{args.stops} stops, {args.rate:g} disruptions/h, {args.hours:g} h")
    print(f"process_time: {info.implementation}, resolution {info.resolution:g} s")
    print()
    print(f"{'kind':<20}{'n':>6}{'cpu med':>10}{'cpu p95':>10}{'cpu max':>10}"
          f"{'wall med':>10}{'wall p95':>10}{'wall max':>10}{'0 cpu':>8}   (ms)")
    for L in summarise(rows):
        print(f"{L['kind']:<20}{L['n']:>6}{L['cpu_med_ms']:>10.2f}{L['cpu_p95_ms']:>10.2f}"
              f"{L['cpu_max_ms']:>10.2f}{L['wall_med_ms']:>10.2f}{L['wall_p95_ms']:>10.2f}"
              f"{L['wall_max_ms']:>10.2f}{L['zero_cpu']:>8}")
    zero = sum(1 for r in rows if r["payload"]["cpu_seconds"] == 0.0)
    if rows and zero / len(rows) > 0.2:
        print(f"WARNING: {zero} of {len(rows)} decisions read 0.0 s CPU: this machine's CPU "
              "timer ticks more coarsely than a replan takes (Windows: ~15.6 ms). Use "
              "wall_seconds for per-replan latency here, and CPU only summed over a run.")
    print()
    run_cpu = totals["run_cpu_s"]
    print(f"whole runs: {run_cpu:.2f} s CPU ({totals['run_wall_s']:.2f} s wall); agents "
          f"{totals['agent_cpu_seconds']:.2f} s, of which logged replans "
          f"{totals['replan_cpu_seconds']:.2f} s and kept {totals['kept_cpu_seconds']:.2f} s; "
          f"coordination hook {totals['coordination_cpu_seconds']:.2f} s")
    if prof is not None:
        stats = pstats.Stats(prof, stream=io.StringIO())
        buckets, total = profile_buckets(stats)
        print(f"\ncProfile of seed 0 (own time, {total:.2f} s; profiling slows the run):")
        for label, sec in sorted(buckets.items(), key=lambda kv: -kv[1]):
            print(f"  {label:<34}{sec:>8.2f} s {100 * sec / total:>6.1f}%")

    if not args.no_csv and rows:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        keys = ["seed", "t", "agent_id", "event", "trigger", "cpu_seconds", "wall_seconds",
                "expanded", "route_found", "gain_s", "mode"]
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in rows:
                p = r["payload"]
                w.writerow([r["seed"], r["t"], r["agent_id"], r["event"], p.get("trigger"),
                            p.get("cpu_seconds"), p.get("wall_seconds"), p.get("expanded"),
                            p.get("route_found"), p.get("gain_s"), p.get("mode")])
        print(f"\nwrote {CSV_PATH} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    main()
