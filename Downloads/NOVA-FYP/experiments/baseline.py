"""Illustrative fixed-distance baseline (not a safety device)."""
def should_alert(distance_m: float | None, threshold_m: float = 2.0) -> bool:
    return distance_m is not None and 0 <= distance_m <= threshold_m
