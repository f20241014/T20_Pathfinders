"""Conflict Detection Module: Translates routes to occupancies and sweeps for capacity overloads."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Node = str
Edge = Tuple[Node, Node]


@dataclass(frozen=True)
class VehicleRoute:
    vehicle_id: str
    nodes: Sequence[Node]
    start_time: float
    travel_times: Dict[Edge, float]
    priority_weight: float = 1.0
    detour_extra_time: float = 5.0


@dataclass(frozen=True)
class RoadOccupancy:
    vehicle_id: str
    edge: Edge
    entry_time: float
    exit_time: float


@dataclass
class Conflict:
    edge: Edge
    vehicles: Set[str]
    overlap_start: float
    overlap_end: float
    earliest_arrival: float

    @property
    def duration(self) -> float:
        return self.overlap_end - self.overlap_start

    def time_to_conflict(self, t_now: float) -> float:
        """TTC = start of earliest conflict minus current time t_now."""
        return self.overlap_start - t_now


def get_edge_time(times: Dict[Edge, float], u: Node, v: Node) -> float:
    if (u, v) in times:
        duration = float(times[(u, v)])
    else:
        raise ValueError(f"Missing travel time for directed edge {u}->{v}")
    if duration <= 0:
        raise ValueError(f"Travel time for {u}->{v} must be positive")
    return duration


def prepare_occupancies(routes: Iterable[VehicleRoute]) -> List[RoadOccupancy]:
    """Convert vehicle routes into timed edge-occupancy intervals [planned_enter, planned_exit]."""
    occupancies: List[RoadOccupancy] = []
    seen_ids: Set[str] = set()

    for route in routes:
        if route.vehicle_id in seen_ids:
            raise ValueError(f"Duplicate vehicle ID: {route.vehicle_id}")
        seen_ids.add(route.vehicle_id)

        if route.start_time < 0:
            raise ValueError("Route start time cannot be negative")
        if len(route.nodes) < 2:
            continue

        current_time = float(route.start_time)
        for u, v in zip(route.nodes[:-1], route.nodes[1:]):
            if u == v:
                raise ValueError("A route cannot contain a self-loop")
            duration = get_edge_time(route.travel_times, u, v)
            occupancies.append(RoadOccupancy(
                vehicle_id=route.vehicle_id,
                edge=(u, v),
                entry_time=current_time,
                exit_time=current_time + duration,
            ))
            current_time += duration

    return occupancies


def _validate_detection_inputs(
    edge_capacities: Dict[Edge, int],
    delta_t: float,
) -> None:
    if delta_t < 0:
        raise ValueError("delta_t cannot be negative")
    for edge, cap in edge_capacities.items():
        if cap < 1:
            raise ValueError(f"Edge capacity for {edge} must be >= 1, got {cap}")


def _effective_clash_window(occ: RoadOccupancy, delta_t: float) -> Tuple[float, float]:
    """
    D6 Arrival-Gap Rule + Physical Occupancy:
    Two vehicles using the same edge clash if their physical occupancies
    [entry, exit) overlap OR their arrival gap |entry_2 - entry_1| < delta_t.
    This corresponds to the effective reservation interval:
    [entry_time, max(exit_time, entry_time + delta_t)).
    """
    return occ.entry_time, max(occ.exit_time, occ.entry_time + delta_t)


def detect_conflicts(
    occupancies: Iterable[RoadOccupancy],
    edge_capacities: Optional[Dict[Edge, int]] = None,
    delta_t: float = 0.0,
) -> List[Conflict]:
    """Sweeps over entry/exit events to detect capacity overloads on directed edges."""
    caps = edge_capacities if edge_capacities is not None else {}
    _validate_detection_inputs(caps, delta_t)

    edge_groups: Dict[Edge, List[RoadOccupancy]] = {}
    for item in occupancies:
        edge_groups.setdefault(item.edge, []).append(item)

    conflicts: List[Conflict] = []

    for edge, users in edge_groups.items():
        k_e = caps.get(edge, 1)

        events = []
        for u in users:
            start_t, end_t = _effective_clash_window(u, delta_t)
            if end_t <= start_t:
                continue
            events.append((start_t, "ENTRY", u))
            events.append((end_t, "EXIT", u))

        # Process EXIT (0) before ENTRY (1) at identical timestamps to avoid 0-duration clashes
        events.sort(key=lambda x: (x[0], 0 if x[1] == "EXIT" else 1))

        active_occupancies: Set[RoadOccupancy] = set()
        in_overload = False
        overload_start = 0.0
        accumulated_vehicles: Set[str] = set()
        accumulated_occupancies: List[RoadOccupancy] = []

        for t, event_type, occ in events:
            if event_type == "ENTRY":
                active_occupancies.add(occ)
                active_vehicles = {o.vehicle_id for o in active_occupancies}
                if len(active_vehicles) > k_e:
                    if not in_overload:
                        in_overload = True
                        overload_start = t
                        accumulated_vehicles = active_vehicles
                        accumulated_occupancies = list(active_occupancies)
                    else:
                        accumulated_vehicles.update(active_vehicles)
                        accumulated_occupancies.extend(active_occupancies)
            else:
                if in_overload:
                    accumulated_vehicles.update(o.vehicle_id for o in active_occupancies)
                    accumulated_occupancies.extend(active_occupancies)
                active_occupancies.discard(occ)
                if in_overload and len({o.vehicle_id for o in active_occupancies}) <= k_e:
                    in_overload = False
                    if t > overload_start:
                        earliest = min(o.entry_time for o in accumulated_occupancies)
                        conflicts.append(Conflict(
                            edge=edge,
                            vehicles=set(accumulated_vehicles),
                            overlap_start=overload_start,
                            overlap_end=t,
                            earliest_arrival=earliest,
                        ))
                    accumulated_vehicles = set()
                    accumulated_occupancies = []

    return sorted(conflicts, key=lambda c: (c.overlap_start, c.edge, sorted(c.vehicles)))


def verify_conflicts_brute_force(
    occupancies: Iterable[RoadOccupancy],
    edge_capacities: Optional[Dict[Edge, int]] = None,
    delta_t: float = 0.0,
) -> List[Conflict]:
    """Brute-force checker that counts active vehicles at every interval boundary."""
    caps = edge_capacities if edge_capacities is not None else {}
    _validate_detection_inputs(caps, delta_t)

    items = list(occupancies)
    conflicts: List[Conflict] = []

    edge_groups: Dict[Edge, List[RoadOccupancy]] = {}
    for item in items:
        edge_groups.setdefault(item.edge, []).append(item)

    for edge, users in edge_groups.items():
        k_e = caps.get(edge, 1)

        effective_windows = []
        for u in users:
            entry = u.entry_time
            exit_t = max(u.exit_time, entry + delta_t)
            if exit_t > entry:
                effective_windows.append((u, entry, exit_t))

        time_points = set()
        for _, entry, exit_t in effective_windows:
            time_points.add(entry)
            time_points.add(exit_t)

        sorted_times = sorted(time_points)
        overload_start: Optional[float] = None
        overload_end: Optional[float] = None
        period_vehicles: Set[str] = set()
        period_occupancies: List[RoadOccupancy] = []

        for t_start, t_end in zip(sorted_times, sorted_times[1:]):
            active = [
                u for u, entry, exit_t in effective_windows
                if entry <= t_start < exit_t
            ]
            if len({u.vehicle_id for u in active}) > k_e:
                if overload_start is None:
                    overload_start = t_start
                overload_end = t_end
                period_vehicles.update(u.vehicle_id for u in active)
                period_occupancies.extend(active)
            elif overload_start is not None:
                assert overload_end is not None
                earliest = min(u.entry_time for u in period_occupancies)
                conflicts.append(Conflict(
                    edge=edge,
                    vehicles=period_vehicles,
                    overlap_start=overload_start,
                    overlap_end=overload_end,
                    earliest_arrival=earliest,
                ))
                overload_start = None
                overload_end = None
                period_vehicles = set()
                period_occupancies = []

        if overload_start is not None:
            assert overload_end is not None
            earliest = min(u.entry_time for u in period_occupancies)
            conflicts.append(Conflict(
                edge=edge,
                vehicles=period_vehicles,
                overlap_start=overload_start,
                overlap_end=overload_end,
                earliest_arrival=earliest,
            ))

    return sorted(conflicts, key=lambda c: (c.overlap_start, c.edge, sorted(c.vehicles)))
