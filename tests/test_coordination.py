"""Comprehensive Unit Tests for R5 Coordination Module."""

import random
import pytest

from src.coordination.conflict import (
    Conflict,
    RoadOccupancy,
    VehicleRoute,
    detect_conflicts,
    prepare_occupancies,
    verify_conflicts_brute_force,
)
from src.coordination.messages import MessageCounter
from src.coordination.negotiation import VirtualCreditBudget, run_vickrey_auction
from src.coordination.trigger import (
    AlwaysTrigger,
    NeverTrigger,
    ThresholdTrigger,
    filter_and_trigger_conflicts,
)


def test_delta_t_arrival_gap_rule():
    """D6 arrival-gap test: arrivals 30s apart on 1-slot road clash iff arrival gap < delta_t."""
    routes = [
        VehicleRoute("V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 5.0}),
        VehicleRoute("V2", ["A", "B"], start_time=30.0, travel_times={("A", "B"): 5.0}),
    ]
    occs = prepare_occupancies(routes)

    # Arrival gap is |30.0 - 0.0| = 30.0s
    assert len(detect_conflicts(occs, delta_t=0.0)) == 0
    assert len(detect_conflicts(occs, delta_t=10.0)) == 0

    res_60 = detect_conflicts(occs, delta_t=60.0)
    assert len(res_60) == 1
    assert res_60[0].vehicles == {"V1", "V2"}
    assert res_60[0].overlap_start == 30.0
    assert res_60[0].overlap_end == 60.0


def test_simultaneous_exit_and_entry_boundary():
    """Boundary check: V1 [0, 5) and V2 [5, 10) at delta_t=0 must not produce a 0-duration clash."""
    occs = [
        RoadOccupancy("V1", ("A", "B"), entry_time=0.0, exit_time=5.0),
        RoadOccupancy("V2", ("A", "B"), entry_time=5.0, exit_time=10.0),
    ]
    sweep_res = detect_conflicts(occs, edge_capacities={("A", "B"): 1}, delta_t=0.0)
    bf_res = verify_conflicts_brute_force(occs, edge_capacities={("A", "B"): 1}, delta_t=0.0)

    assert sweep_res == []
    assert bf_res == []


def test_invalid_inputs_rejected():
    """Negative delta_t and non-positive edge capacities must raise ValueError."""
    occs = [RoadOccupancy("V1", ("A", "B"), entry_time=0.0, exit_time=5.0)]
    with pytest.raises(ValueError):
        detect_conflicts(occs, delta_t=-1.0)
    with pytest.raises(ValueError):
        detect_conflicts(occs, edge_capacities={("A", "B"): 0})


def test_should_negotiate_3_arguments():
    """Tests should_negotiate with 3 arguments on ongoing and upcoming conflicts."""
    trigger = ThresholdTrigger(tau=60.0)
    routes = {
        "V1": VehicleRoute("V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 50.0}),
        "V2": VehicleRoute("V2", ["A", "B"], start_time=10.0, travel_times={("A", "B"): 50.0}),
    }
    r1 = routes["V1"]
    r2 = routes["V2"]

    assert trigger.should_negotiate(r1, routes, t=0.0) is True
    assert trigger.should_negotiate(r2, routes, t=0.0) is True
    assert trigger.should_negotiate(r1, routes, t=20.0) is True

    trigger_narrow = ThresholdTrigger(tau=5.0)
    assert trigger_narrow.should_negotiate(r1, routes, t=0.0) is False


def test_threshold_trigger_uses_delta_t_and_edge_capacities():
    """ThresholdTrigger uses its delta_t and capacity settings when given three arguments."""
    routes = {
        "V1": VehicleRoute("V1", ["A", "B"], 0.0, {("A", "B"): 5.0}),
        "V2": VehicleRoute("V2", ["A", "B"], 30.0, {("A", "B"): 5.0}),
    }

    assert ThresholdTrigger(tau=31.0).should_negotiate(
        routes["V1"], routes, t=0.0
    ) is False
    assert ThresholdTrigger(tau=31.0, delta_t=10.0).should_negotiate(
        routes["V1"], routes, t=0.0
    ) is False
    assert ThresholdTrigger(tau=31.0, delta_t=60.0).should_negotiate(
        routes["V1"], routes, t=0.0
    ) is True

    three_routes = {
        f"V{i}": VehicleRoute(f"V{i}", ["A", "B"], float(i), {("A", "B"): 10.0})
        for i in range(3)
    }
    assert ThresholdTrigger(tau=20.0, edge_capacities={("A", "B"): 2}).should_negotiate(
        three_routes["V0"], three_routes, t=0.0
    ) is True
    assert ThresholdTrigger(tau=20.0, edge_capacities={("A", "B"): 3}).should_negotiate(
        three_routes["V0"], three_routes, t=0.0
    ) is False


def test_always_trigger_in_and_out_of_conflict():
    """AlwaysTrigger fires only for vehicles involved in a conflict."""
    trigger = AlwaysTrigger()
    routes = {
        "V1": VehicleRoute("V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0}),
        "V2": VehicleRoute("V2", ["A", "B"], start_time=2.0, travel_times={("A", "B"): 10.0}),
        "V3": VehicleRoute("V3", ["C", "D"], start_time=0.0, travel_times={("C", "D"): 10.0}),
    }
    assert trigger.should_negotiate(routes["V1"], routes, t=0.0) is True
    assert trigger.should_negotiate(routes["V2"], routes, t=0.0) is True
    assert trigger.should_negotiate(routes["V3"], routes, t=0.0) is False


def test_random_cross_check_detect_conflicts():
    """500 random trial cross-check comparing edges, vehicle sets, start and end."""
    random.seed(42)
    nodes = ["A", "B", "C", "D"]

    for trial in range(500):
        occupancies = []
        num_vehicles = random.randint(2, 6)
        for i in range(num_vehicles):
            u, v = random.sample(nodes, 2)
            entry = float(random.randint(0, 20))
            dur = float(random.randint(5, 15))
            occupancies.append(RoadOccupancy(
                vehicle_id=f"V{i}",
                edge=(u, v),
                entry_time=entry,
                exit_time=entry + dur,
            ))

        caps = {("A", "B"): random.randint(1, 3), ("B", "C"): random.randint(1, 3)}
        dt = random.choice([0.0, 10.0, 25.0])
        sweep_res = detect_conflicts(occupancies, edge_capacities=caps, delta_t=dt)
        bf_res = verify_conflicts_brute_force(occupancies, edge_capacities=caps, delta_t=dt)

        assert len(sweep_res) == len(bf_res), f"Mismatch count in trial {trial}"
        for s_c, b_c in zip(sweep_res, bf_res):
            assert s_c.edge == b_c.edge, f"Mismatch edge in trial {trial}"
            assert s_c.vehicles == b_c.vehicles, f"Mismatch vehicles in trial {trial}"
            assert abs(s_c.overlap_start - b_c.overlap_start) < 1e-6
            assert abs(s_c.overlap_end - b_c.overlap_end) < 1e-6


def test_three_vehicles_on_two_slot_road():
    """Specific check for 3 vehicles on a 2-slot road."""
    routes = [
        VehicleRoute("V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 20.0}),
        VehicleRoute("V2", ["A", "B"], start_time=5.0, travel_times={("A", "B"): 20.0}),
        VehicleRoute("V3", ["A", "B"], start_time=10.0, travel_times={("A", "B"): 20.0}),
    ]
    occs = prepare_occupancies(routes)
    caps = {("A", "B"): 2}

    sweep_res = detect_conflicts(occs, edge_capacities=caps)
    bf_res = verify_conflicts_brute_force(occs, edge_capacities=caps)

    assert len(sweep_res) == 1
    assert len(bf_res) == 1
    assert sweep_res == bf_res
    assert sweep_res[0].edge == ("A", "B")
    assert sweep_res[0].vehicles == {"V1", "V2", "V3"}
    assert sweep_res[0].overlap_start == 10.0
    assert sweep_res[0].overlap_end == 20.0


def test_same_vehicle_occupancies_do_not_consume_multiple_slots():
    """Multiple overlapping occupancy records for one vehicle count as one vehicle."""
    occs = [
        RoadOccupancy("V1", ("A", "B"), 0.0, 10.0),
        RoadOccupancy("V1", ("A", "B"), 1.0, 9.0),
        RoadOccupancy("V2", ("A", "B"), 1.0, 10.0),
    ]
    capacities = {("A", "B"): 2}

    assert detect_conflicts(occs, edge_capacities=capacities) == []
    assert verify_conflicts_brute_force(occs, edge_capacities=capacities) == []


def test_credit_bounds_and_payment_failure():
    """1 credit and price 6 leaves credits at 1 with no winner."""
    r1 = VehicleRoute(
        "V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0},
        priority_weight=2.0, detour_extra_time=5.0,
    )
    r2 = VehicleRoute(
        "V2", ["A", "B"], start_time=1.0, travel_times={("A", "B"): 10.0},
        priority_weight=2.0, detour_extra_time=3.0,
    )
    budgets = {
        "V1": VirtualCreditBudget("V1", credits=1.0),
        "V2": VirtualCreditBudget("V2", credits=50.0),
    }

    winner, losers, price = run_vickrey_auction([r1, r2], budgets)
    assert winner is None
    assert set(losers) == {"V1", "V2"}
    assert price == 0.0
    assert budgets["V1"].credits == 1.0


def test_vickrey_tie_breaking_lowest_id():
    """Equal bids must deterministically award the slot to the lexicographically lowest ID."""
    r2 = VehicleRoute(
        "V2", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0},
        priority_weight=2.0, detour_extra_time=5.0,
    )
    r1 = VehicleRoute(
        "V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0},
        priority_weight=2.0, detour_extra_time=5.0,
    )
    budgets = {
        "V1": VirtualCreditBudget("V1", credits=50.0),
        "V2": VirtualCreditBudget("V2", credits=50.0),
    }
    winner, losers, price = run_vickrey_auction([r2, r1], budgets)
    assert winner == "V1"
    assert losers == ["V2"]
    assert price == 10.0


def test_truthfulness_sanity_check_under_budgets():
    """Truthful bids are not worse across binding and non-binding credit budgets."""
    credit_levels = [0.5, 3.0, 10.0, 100.0]
    true_value = 5.0
    deviations = [0.0, 2.0, 4.0, 6.0, 9.0, 12.0]

    for credits in credit_levels:
        for competing_bid in (4.0, 8.0):
            rival = VehicleRoute(
                "V2", ["A", "B"], 0.0, {("A", "B"): 10.0},
                priority_weight=competing_bid, detour_extra_time=1.0,
            )

            def payoff(reported_value):
                bidder = VehicleRoute(
                    "V1", ["A", "B"], 0.0, {("A", "B"): 10.0},
                    priority_weight=reported_value, detour_extra_time=1.0,
                )
                budgets = {
                    "V1": VirtualCreditBudget("V1", credits),
                    "V2": VirtualCreditBudget("V2", 100.0),
                }
                winner, _, price = run_vickrey_auction([bidder, rival], budgets)
                return true_value - price if winner == "V1" else 0.0

            truthful_payoff = payoff(true_value)
            for reported_value in deviations:
                assert truthful_payoff >= payoff(reported_value) - 1e-6, (
                    f"Truthful payoff was worse with credits={credits}, "
                    f"competing bid={competing_bid}, deviation={reported_value}"
                )


def test_nobody_can_pay():
    """Test auction when no vehicle has enough credits to pay."""
    r1 = VehicleRoute(
        "V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0},
        priority_weight=2.0, detour_extra_time=5.0,
    )
    r2 = VehicleRoute(
        "V2", ["A", "B"], start_time=1.0, travel_times={("A", "B"): 10.0},
        priority_weight=2.0, detour_extra_time=4.0,
    )
    budgets = {
        "V1": VirtualCreditBudget("V1", credits=2.0),
        "V2": VirtualCreditBudget("V2", credits=2.0),
    }
    winner, losers, price = run_vickrey_auction([r1, r2], budgets)
    assert winner is None
    assert set(losers) == {"V1", "V2"}
    assert price == 0.0


def test_table_driven_threshold_trigger():
    """Table-driven test verifying ThresholdTrigger fires iff TTC < tau."""
    trigger = ThresholdTrigger(tau=30.0)
    test_cases = [
        (10.0, True),
        (0.0, True),
        (-15.0, True),
        (30.0, False),
        (45.0, False),
    ]

    for ttc_val, expected in test_cases:
        c = Conflict(
            edge=("A", "B"),
            vehicles={"V1", "V2"},
            overlap_start=50.0,
            overlap_end=60.0,
            earliest_arrival=0.0,
        )
        t_now = 50.0 - ttc_val
        route = VehicleRoute("V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0})
        routes = {"V1": route}

        result = trigger.should_negotiate(route, routes, t_now, conflict=c)
        assert result is expected, f"Failed for TTC={ttc_val}"


def test_never_trigger_zero_messages():
    """Verifies NeverTrigger yields exactly zero trigger events."""
    trigger = NeverTrigger()
    routes = {
        "V1": VehicleRoute("V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0}),
        "V2": VehicleRoute("V2", ["A", "B"], start_time=2.0, travel_times={("A", "B"): 10.0}),
    }
    occs = prepare_occupancies(routes.values())
    conflicts = detect_conflicts(occs)

    events = filter_and_trigger_conflicts(conflicts, trigger, routes, t_now=0.0)
    assert len(events) == 0


def test_message_counter_d7_and_allocations():
    """Verifies 2 vehicles produce 6 negotiation messages and tracks allocations separately."""
    routes = [
        VehicleRoute(
            "V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0},
            priority_weight=2.0,
        ),
        VehicleRoute(
            "V2", ["A", "B"], start_time=1.0, travel_times={("A", "B"): 10.0},
            priority_weight=1.0,
        ),
    ]
    budgets = {"V1": VirtualCreditBudget("V1", 50.0), "V2": VirtualCreditBudget("V2", 50.0)}
    counter = MessageCounter()

    run_vickrey_auction(routes, budgets, counter)
    assert counter.requests == 2
    assert counter.responses == 2
    assert counter.results == 2
    assert counter.total_negotiation_messages == 6

    counter.log_message("allocation")
    assert counter.allocations == 1
    assert counter.total_negotiation_messages == 6
    assert counter.total_messages == 7


def test_input_list_order_independence():
    """Verifies auction results do not depend on the order of input contesting_routes list."""
    r1 = VehicleRoute(
        "V1", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0},
        priority_weight=2.0,
    )
    r2 = VehicleRoute(
        "V2", ["A", "B"], start_time=0.0, travel_times={("A", "B"): 10.0},
        priority_weight=1.0,
    )

    b1 = {"V1": VirtualCreditBudget("V1", 50.0), "V2": VirtualCreditBudget("V2", 50.0)}
    b2 = {"V1": VirtualCreditBudget("V1", 50.0), "V2": VirtualCreditBudget("V2", 50.0)}

    res1 = run_vickrey_auction([r1, r2], b1)
    res2 = run_vickrey_auction([r2, r1], b2)

    assert res1[0] == res2[0]
    assert res1[2] == res2[2]
