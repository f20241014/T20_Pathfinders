"""Trigger Policies Module: Evaluates when conflicts warrant negotiation."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from src.coordination.conflict import Conflict, VehicleRoute, detect_conflicts, prepare_occupancies

Edge = Tuple[str, str]


@dataclass
class ConflictTriggerEvent:
    event_type: str
    edge: Edge
    vehicles: Tuple[str, ...]
    trigger_time: float
    overlap_end: float
    ttc: float


class TriggerPolicy:
    """Common interface for coordination triggers matching project plan."""
    def should_negotiate(
        self,
        vehicle: VehicleRoute,
        routes: Dict[str, VehicleRoute],
        t: float,
        conflict: Optional[Conflict] = None,
    ) -> bool:
        raise NotImplementedError


class NeverTrigger(TriggerPolicy):
    """Independent baseline: never negotiates."""
    def should_negotiate(
        self,
        vehicle: VehicleRoute,
        routes: Dict[str, VehicleRoute],
        t: float,
        conflict: Optional[Conflict] = None,
    ) -> bool:
        return False


class AlwaysTrigger(TriggerPolicy):
    """Ablation condition: always negotiates if a conflict exists."""
    def __init__(
        self,
        delta_t: float = 0.0,
        edge_capacities: Optional[Dict[Edge, int]] = None,
    ):
        self.delta_t = delta_t
        self.edge_capacities = edge_capacities if edge_capacities is not None else {}

    def should_negotiate(
        self,
        vehicle: VehicleRoute,
        routes: Dict[str, VehicleRoute],
        t: float,
        conflict: Optional[Conflict] = None,
    ) -> bool:
        if conflict is not None:
            return True
        occs = prepare_occupancies(routes.values())
        conflicts = detect_conflicts(
            occs,
            edge_capacities=self.edge_capacities,
            delta_t=self.delta_t,
        )
        return any(vehicle.vehicle_id in c.vehicles for c in conflicts)


class ThresholdTrigger(TriggerPolicy):
    """Proposed system: negotiates iff TTC < tau (including negative TTC for ongoing conflicts)."""
    def __init__(
        self,
        tau: float,
        delta_t: float = 0.0,
        edge_capacities: Optional[Dict[Edge, int]] = None,
    ):
        self.tau = tau
        self.delta_t = delta_t
        self.edge_capacities = edge_capacities if edge_capacities is not None else {}

    def should_negotiate(
        self,
        vehicle: VehicleRoute,
        routes: Dict[str, VehicleRoute],
        t: float,
        conflict: Optional[Conflict] = None,
    ) -> bool:
        if conflict is None:
            occs = prepare_occupancies(routes.values())
            conflicts = detect_conflicts(
                occs,
                edge_capacities=self.edge_capacities,
                delta_t=self.delta_t,
            )
            v_conflicts = [c for c in conflicts if vehicle.vehicle_id in c.vehicles]
            if not v_conflicts:
                return False
            return any(c.time_to_conflict(t) < self.tau for c in v_conflicts)

        ttc = conflict.time_to_conflict(t)
        return ttc < self.tau


def filter_and_trigger_conflicts(
    conflicts: Iterable[Conflict],
    trigger_policy: TriggerPolicy,
    routes_dict: Dict[str, VehicleRoute],
    t_now: float,
) -> List[ConflictTriggerEvent]:
    """Generate negotiation trigger events for conflicts satisfying the policy."""
    events: List[ConflictTriggerEvent] = []
    for c in conflicts:
        ttc = c.time_to_conflict(t_now)
        any_triggered = False
        for v_id in c.vehicles:
            if v_id in routes_dict and trigger_policy.should_negotiate(
                routes_dict[v_id], routes_dict, t_now, conflict=c
            ):
                any_triggered = True
                break

        if any_triggered:
            events.append(ConflictTriggerEvent(
                event_type="ROAD_SLOT_NEGOTIATION_TRIGGER",
                edge=c.edge,
                vehicles=tuple(sorted(c.vehicles)),
                trigger_time=t_now,
                overlap_end=c.overlap_end,
                ttc=ttc,
            ))
    return events
