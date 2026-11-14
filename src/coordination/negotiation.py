"""Negotiation & Game Theory Module: Implements Vickrey Second-Price Auctions."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.coordination.conflict import VehicleRoute
from src.coordination.messages import MessageCounter


@dataclass
class VirtualCreditBudget:
    """Budget tracker to prevent single vehicle dominance. Credits never drop below 0."""
    vehicle_id: str
    credits: float = 100.0


def calculate_delay_cost(
    route: VehicleRoute,
    world_edge_time_fn: Optional[callable] = None,
    k_shortest_paths_fn: Optional[callable] = None,
) -> float:
    """
    Computes delay cost bid: priority_weight * extra_time.
    Uses fallback detour_extra_time when map routing helper is not passed.
    """
    extra_time = route.detour_extra_time
    return route.priority_weight * extra_time


def run_vickrey_auction(
    contesting_routes: List[VehicleRoute],
    budgets: Dict[str, VirtualCreditBudget],
    msg_counter: Optional[MessageCounter] = None,
) -> Tuple[Optional[str], List[str], float]:
    """
    Vickrey Auction Primitive:
    - Bids: delay cost. Vehicle cannot win if its credit balance is below the price.
    - Tie-breaking: lowest vehicle_id.
    - Message counts (D7): 1 result message sent per involved vehicle.
    """
    if not contesting_routes:
        return None, [], 0.0

    m_agents = len(contesting_routes)
    if msg_counter:
        for _ in range(m_agents):
            msg_counter.log_message("negotiation_request")
            msg_counter.log_message("negotiation_response")
            msg_counter.log_message("negotiation_result")

    bids: Dict[str, float] = {}
    for r in contesting_routes:
        avail_credits = budgets[r.vehicle_id].credits if r.vehicle_id in budgets else 0.0
        if avail_credits > 0:
            bids[r.vehicle_id] = calculate_delay_cost(r)
        else:
            bids[r.vehicle_id] = 0.0

    sorted_candidates = sorted(
        contesting_routes,
        key=lambda r: (-bids[r.vehicle_id], r.vehicle_id),
    )

    winner = sorted_candidates[0]
    highest_bid = bids[winner.vehicle_id]

    if len(sorted_candidates) > 1:
        second_highest_bid = bids[sorted_candidates[1].vehicle_id]
    else:
        second_highest_bid = 0.0

    winner_budget = budgets.get(winner.vehicle_id)

    if (
        winner_budget is None
        or winner_budget.credits < second_highest_bid
        or highest_bid == 0.0
    ):
        losers = [r.vehicle_id for r in contesting_routes]
        return None, losers, 0.0

    winner_budget.credits = max(0.0, winner_budget.credits - second_highest_bid)
    losers = [r.vehicle_id for r in contesting_routes if r.vehicle_id != winner.vehicle_id]
    return winner.vehicle_id, losers, second_highest_bid
