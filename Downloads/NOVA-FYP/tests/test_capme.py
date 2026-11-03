import pytest
from capme.engine import Event, classify_event

def test_critical_override():
    e = Event("step", 0, 0, 0, 0, 0, confirmed_critical=True)
    assert classify_event(e).priority == "CRITICAL"

def test_low_priority():
    assert classify_event(Event("chair", 0, 0, 0, 0, 0)).priority == "SUPPRESSED"

def test_high_priority():
    assert classify_event(Event("hazard", 1, 1, 1, 1, 1)).priority == "CRITICAL"

def test_invalid_input():
    with pytest.raises(ValueError):
        classify_event(Event("invalid", -1, 0, 0, 0, 0))
