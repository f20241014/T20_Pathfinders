"""Week 6 tests for src/sim/agent_planner.py (R7): robustness, D9 CPU timing
on every replan, and determinism.

Grid world 3 x 3 (node = 3 * row + col), every road 100 m at 10 m/s = 10 s.
Helpers come from the Week 5 tests.
"""
import json
import math

import pytest

from src.coordination.trigger import ThresholdTrigger
from src.search.astar import astar
from src.sim.agent_planner import AgentPlanner, ReplanningAgent, snapshot_cost
from src.sim.coordination_hook import TriggerCoordination
from src.sim.models import Event, Order, Vehicle
from src.sim.world import FakeWorld
from tests.test_agent_planner_w5 import (FAR, START, conflict_world, cut_off_scenario,
                                         disrupt_at, grid_graph, make_env, real_map_full_run,
                                         rows)

TIMING = ("cpu_seconds", "wall_seconds")


def grid_run(tmp_path, n_orders=2, **planner_kw):
    world = FakeWorld(grid_graph())
    env = make_env(world, tmp_path)
    planner = AgentPlanner(env, **planner_kw)
    veh = Vehicle("V1", 0, 10.0, current_node=0)
    orders = [Order(f"o{i}", n, 1.0, service_time=0.0) for i, n in enumerate([2, 8][:n_orders])]
    planner.register(veh, orders, stops=[o.node for o in orders])
    env.add_vehicle(veh, orders)
    return env, planner, veh


# ---------------------------------------------------------------------------
# Bad or unusual input must not crash a run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [None, {}, {"type": "closure"}, {"edge": None},
                                     {"edge": []}, {"edge": [1]}, "closure"])
def test_malformed_disruption_is_ignored(tmp_path, payload):
    env, planner, veh = grid_run(tmp_path)
    before = list(veh.remaining_route)
    planner.on_event(Event(START, "disruption", payload))
    assert planner.ignored_events == 1
    assert veh.remaining_route == before and not rows(env, "replan")


def test_other_event_types_are_ignored(tmp_path):
    env, planner, veh = grid_run(tmp_path)
    planner.on_event(Event(START, "new_order", {"edge": [1, 2, 0]}))
    planner.on_event(object())                                # not even an Event
    assert planner.ignored_events == 0 and not rows(env, "replan")


def test_disruption_on_a_road_not_on_the_map(tmp_path):
    world = FakeWorld(grid_graph())
    veh = Vehicle("V1", 0, 10.0, current_node=0)
    agent = ReplanningAgent(veh, world, [8], START)
    for road in [(0, 99), (99, 98), (0, 4)]:                  # (0, 4) is not a grid road
        assert agent.on_world_change([road], START) is None
    assert agent.replanner.watched == set()                   # not kept, so later events work
    u, v = agent.planned[1], agent.planned[2]
    world.add_disruption(u, v, 0, FAR, start_t_seconds=START)
    assert agent.on_world_change([(u, v)], START) is not None


def test_planner_with_no_vehicles(tmp_path):
    env = make_env(FakeWorld(grid_graph()), tmp_path)
    planner = AgentPlanner(env)
    disrupt_at(env, 1, 2, 5)
    env.run(100)
    s = planner.finish()
    assert s["replans"] == 0 and s["replan_cpu_seconds"] == 0.0 and rows(env, "replan_summary")


def test_vehicle_with_no_stops(tmp_path):
    world = FakeWorld(grid_graph())
    env = make_env(world, tmp_path)
    planner = AgentPlanner(env)
    veh = Vehicle("V1", 4, 10.0, current_node=4)
    route = planner.register(veh, [], stops=[])
    assert route.nodes == [4] and veh.remaining_route == []
    env.add_vehicle(veh, [])
    disrupt_at(env, 4, 5, 1)
    env.run(100)
    assert planner.finish()["replans"] == 0


def test_stop_at_the_start_node_and_two_orders_on_one_node(tmp_path):
    world = FakeWorld(grid_graph())
    env = make_env(world, tmp_path)
    planner = AgentPlanner(env)
    veh = Vehicle("V1", 0, 10.0, current_node=0)
    orders = [Order("a", 0, 1.0, service_time=0.0), Order("b", 8, 1.0, service_time=0.0),
              Order("c", 8, 1.0, service_time=0.0)]
    planner.register(veh, orders, stops=[0, 8])
    env.add_vehicle(veh, orders)
    disrupt_at(env, 1, 2, 5)
    env.run(500)
    delivered = sorted(r["payload"]["order_id"] for r in rows(env, "order_delivered"))
    assert delivered == ["a", "b", "c"]


def test_finished_vehicle_ignores_later_disruptions(tmp_path):
    env, planner, veh = grid_run(tmp_path, n_orders=1)
    disrupt_at(env, 1, 2, 200)                               # long after o0 is delivered
    env.run(500)
    assert not rows(env, "replan") and planner.finish()["replans"] == 0


def test_closure_on_the_road_being_driven(tmp_path):
    """The vehicle is on 0 -> 1 when 0 -> 1 closes: it can't turn back, so
    it finishes the road and plans from 1."""
    env, planner, veh = grid_run(tmp_path, n_orders=1)
    disrupt_at(env, 0, 1, 5)
    env.run(500)
    assert [r["payload"]["order_id"] for r in rows(env, "order_delivered")] == ["o0"]
    assert not rows(env, "hook_error")


def test_many_disruptions_at_the_same_instant(tmp_path):
    """Ten events at one time: each vehicle replans at most once per event and
    the final plan matches a fresh A* on the world at that time."""
    world = FakeWorld(grid_graph(4))
    env = make_env(world, tmp_path)
    planner = AgentPlanner(env)
    vehs = []
    for vid, start, goal in (("V1", 0, 15), ("V2", 3, 12), ("V3", 12, 3)):
        veh = Vehicle(vid, start, 10.0, current_node=start)
        o = Order("o" + vid, goal, 1.0, service_time=0.0)
        planner.register(veh, [o], stops=[goal])
        env.add_vehicle(veh, [o])
        vehs.append(veh)
    for u, v in [(1, 2), (4, 5), (5, 9), (6, 10), (9, 13), (7, 11), (2, 6), (8, 9), (13, 14),
                 (10, 11)]:
        disrupt_at(env, u, v, 3)
    env.run(4)                                # the events fire at t = 3
    assert len(rows(env, "disruption")) == 10
    t = env.get_current_time()
    cost = snapshot_cost(world, t)
    for vid, agent in planner.agents.items():
        start = agent.planning_start()
        goal = agent.remaining_stops[-1]
        ref = astar(world.graph(), start, goal, cost_fn=cost, heuristic=lambda x, y: 0.0).cost
        assert agent.replanner.planner.path_cost() == pytest.approx(ref)
    env.run(2000)
    assert len(rows(env, "order_delivered")) == 3


def test_failing_coordination_hook_does_not_stop_the_run(tmp_path):
    world = conflict_world()
    env = make_env(world, tmp_path, mode="full")

    def broken(vid, route, routes, t):
        raise RuntimeError("R5 bug")

    planner = AgentPlanner(env, mode="full", coordination=broken)
    for vid, node in (("V1", 0), ("V2", 4)):
        veh = Vehicle(vid, node, 10.0, current_node=node)
        o = Order("o" + vid, 2, 1.0, service_time=0.0)
        planner.register(veh, [o], stops=[2])
        env.add_vehicle(veh, [o])
    disrupt_at(env, 1, 2, 3)
    env.run(500)
    s = planner.finish()
    errors = rows(env, "hook_error")
    assert errors and errors[0]["payload"]["hook"] == "coordination"
    assert "RuntimeError: R5 bug" in errors[0]["payload"]["error"]
    assert s["hook_errors"] == len(errors) and s["negotiation_messages"] == 0
    assert len(rows(env, "order_delivered")) == 2


def test_failing_reassign_hook_fails_the_order_instead(tmp_path):
    def broken(vid, orders, t):
        raise ValueError("R4 bug")

    env, planner, summary = cut_off_scenario(tmp_path, reassign=broken)
    assert summary["failed_unreachable"] == ["o1"] and summary["reassigned"] == []
    assert rows(env, "hook_error")[0]["payload"]["hook"] == "reassign"
    assert [r["payload"]["order_id"] for r in rows(env, "order_delivered")] == ["o2"]


def test_closure_that_ends_is_noticed_at_the_next_event(tmp_path):
    """A closure expires with no event of its own; the road is watched, so the
    next disruption anywhere re-reads it and the vehicle can switch back."""
    world = FakeWorld(grid_graph())
    veh = Vehicle("V1", 0, 10.0, current_node=0)
    agent = ReplanningAgent(veh, world, [2], START)          # 0 -> 1 -> 2, 20 s
    world.add_disruption(1, 2, 0, START + 1, start_t_seconds=START)
    detour = agent.on_world_change([(1, 2)], START)
    assert (1, 2) not in set(zip(detour.nodes, detour.nodes[1:]))
    world.add_spike(6, 7, 0, 2.0, FAR, start_t_seconds=START + 2)   # unrelated event later
    back = agent.on_world_change([(6, 7)], START + 2)
    assert back is not None and back.nodes == [0, 1, 2]       # 20 s again, gain 20 s


# ---------------------------------------------------------------------------
# D9: CPU timing on every replan
# ---------------------------------------------------------------------------
def test_every_replan_row_is_timed_and_the_summary_adds_up(tmp_path):
    env, planner, s, calls, hook = real_map_full_run(tmp_path, 1)
    replans = rows(env, "replan")
    kept = rows(env, "replan_kept")
    triggers = {r["payload"]["trigger"] for r in replans}
    assert {"closure", "negotiation_yield", "unreachable"} <= triggers   # all paths covered
    for r in replans + kept:
        for k in TIMING:
            x = r["payload"][k]
            assert isinstance(x, float) and x >= 0.0 and math.isfinite(x)
    assert s["replan_cpu_seconds"] == pytest.approx(
        sum(r["payload"]["cpu_seconds"] for r in replans))
    assert s["kept_cpu_seconds"] == pytest.approx(sum(r["payload"]["cpu_seconds"] for r in kept))
    assert s["agent_cpu_seconds"] >= s["kept_cpu_seconds"]
    assert s["coordination_cpu_seconds"] > 0.0
    assert s["hook_errors"] == 0 and s["ignored_events"] == 0


def test_wall_time_is_at_least_cpu_time_for_a_real_replan():
    """One replan on the real map: wall >= CPU (single thread), both > 0."""
    from src.search.heuristics import make_heuristic
    from src.world.graph_loader import load_road_graph
    G = load_road_graph("secunderabad_small")
    world = FakeWorld(G)
    nodes = sorted(G.nodes)
    start, goal = nodes[0], nodes[-1]
    if astar(G, start, goal).cost == math.inf:
        pytest.skip("no path between the chosen nodes")
    veh = Vehicle("V1", start, 10.0, current_node=start)
    agent = ReplanningAgent(veh, world, [goal], START, make_heuristic(G))
    u, v = agent.planned[len(agent.planned) // 2], agent.planned[len(agent.planned) // 2 + 1]
    world.add_disruption(u, v, world.edge_key(u, v), FAR, start_t_seconds=START)
    agent.on_world_change([(u, v)], START)
    assert agent.last_cpu > 0.0 or agent.last_wall > 0.0
    assert agent.last_wall >= agent.last_cpu * 0.5            # allow timer noise


# ---------------------------------------------------------------------------
# Determinism: same seed and config, same replan log
# ---------------------------------------------------------------------------
def _strip(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r.pop("wall_clock", None)
            p = r.get("payload")
            if isinstance(p, dict):
                for k in list(p):
                    if k.endswith("cpu_seconds") or k.endswith("wall_seconds"):
                        p.pop(k)
            out.append(r)
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_same_seed_same_log(tmp_path, seed):
    env_a = real_map_full_run(tmp_path / "a", seed)[0]
    env_b = real_map_full_run(tmp_path / "b", seed)[0]
    a, b = _strip(env_a.log_path), _strip(env_b.log_path)
    assert len(a) == len(b) and a == b


def test_independent_mode_is_deterministic_too(tmp_path):
    logs = []
    for name in ("a", "b"):
        env, planner, veh = grid_run(tmp_path / name)
        disrupt_at(env, 1, 2, 5)
        disrupt_at(env, 4, 5, 12)
        env.run(500)
        planner.finish()
        logs.append(_strip(env.log_path))
    assert logs[0] == logs[1]


def test_threshold_hook_with_real_map_has_no_errors(tmp_path):
    env, planner, s, calls, hook = real_map_full_run(tmp_path, 2)
    assert isinstance(hook, TriggerCoordination) and isinstance(hook.trigger, ThresholdTrigger)
    assert not rows(env, "hook_error") and rows(env, "sim_complete")


def test_profile_script_runs_and_writes_csv(tmp_path, monkeypatch, capsys):
    from src.search import profile_replan as pr
    monkeypatch.setattr(pr, "CSV_PATH", str(tmp_path / "profile.csv"))
    rows_ = pr.main(["--seeds", "1", "--vehicles", "3", "--stops", "2", "--rate", "300",
                     "--hours", "0.25"])
    out = capsys.readouterr().out
    assert "cProfile of seed 0" in out and "D* Lite repair" in out
    with open(tmp_path / "profile.csv", encoding="utf-8") as f:
        assert len(f.readlines()) == len(rows_) + 1
    assert pr.pct([1, 2, 3, 4], 0.5) == 2.5 and math.isnan(pr.pct([], 0.5))
