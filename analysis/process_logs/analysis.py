import json
from collections import namedtuple

Event = namedtuple('Event', [
    'timestamp',
    'name',
    'params',
]) 

def load_events(log_path, num_lines=-1):
    events = []
    lines_read = 0
    with open(log_path) as f:
        for line in f:
            # Skip lines that are not events.
            if not line.startswith("event | "):
                continue

            if num_lines > 0 and lines_read > num_lines:
                break

            _, timestamp, event, params = line.strip().split(" | ")
            events.append((float(timestamp), event, json.loads(params)))

            lines_read += 1

    # Sort events by timestamp.
    events.sort(key=lambda x: x[0])

    # Adjust timestamps to start at 0.
    start_time = events[0][0]
    print(f"Start time: {start_time}")

    return [
        Event(
            timestamp=timestamp - start_time,
            name=event,
            params=params,
        )
        for timestamp, event, params in events
    ]

def filter_events(events, event_name):
    return [event for event in events if event.name == event_name]