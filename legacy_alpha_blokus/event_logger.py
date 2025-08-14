import time
import json

def log_event(event_name, parameters={}):
    print(
        f"event | {time.time()} | {event_name} | {json.dumps(parameters)}"
    )