"""Experimental CAPME baseline; not safety-certified."""
from dataclasses import dataclass
from typing import Literal

Priority = Literal["CRITICAL", "NAVIGATION", "SUPPRESSED"]

@dataclass(frozen=True)
class Event:
    label: str
    risk: float             # normalized 0..1
    proximity: float        # normalized 0..1 (higher = closer)
    movement: float         # normalized 0..1 (higher = more concerning)
    path_relevance: float   # normalized 0..1
    destination_relevance: float  # normalized 0..1
    confirmed_critical: bool = False

@dataclass(frozen=True)
class Decision:
    label: str
    score: float
    priority: Priority
    reason: str

WEIGHTS = {"risk": 0.35, "proximity": 0.25, "movement": 0.15,
           "path_relevance": 0.15, "destination_relevance": 0.10}

def classify_event(event: Event) -> Decision:
    values = (event.risk, event.proximity, event.movement,
              event.path_relevance, event.destination_relevance)
    if not all(0 <= v <= 1 for v in values):
        raise ValueError("All normalized features must be between 0 and 1")
    score = round(100 * sum(getattr(event, name) * weight
                            for name, weight in WEIGHTS.items()), 2)
    # A confirmed safety-critical hazard bypasses the experimental score.
    if event.confirmed_critical:
        return Decision(event.label, score, "CRITICAL", "Explicit safety override")
    if score >= 70:
        return Decision(event.label, score, "CRITICAL", "High experimental score")
    if score >= 40:
        return Decision(event.label, score, "NAVIGATION", "Moderate experimental score")
    return Decision(event.label, score, "SUPPRESSED", "Low experimental score")
