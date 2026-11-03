from capme.engine import Event, classify_event

samples = [
    Event("chair left of path", .2, .8, .1, .1, .0),
    Event("destination doorway", .2, .5, .0, .8, 1.0),
    Event("confirmed downward step", 1, .9, .0, 1, .0, True),
]
for event in samples:
    result = classify_event(event)
    print(f"{result.label}: {result.priority} (score={result.score}) — {result.reason}")
