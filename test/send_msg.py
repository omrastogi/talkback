"""Send a message to Robin the way the ambient-reminder service does.

    python test/send_msg.py "Time to take your evening medication."
    python test/send_msg.py "Time for your walk." "You have been sitting for two hours."
    CA_ENDPOINT=https://<tunnel>/proactive python test/send_msg.py "hello"
"""
import os
import sys
import uuid
import requests
from datetime import datetime, timezone

# The public tunnel in front of the Robin server. Override with CA_ENDPOINT for a local run
# (http://127.0.0.1:8000/proactive), or when the quick tunnel is restarted and renamed.
DEFAULT_ENDPOINT = "https://hundreds-midnight-glory-gba.trycloudflare.com/proactive"


def _post_to_ca(endpoint_id: str, utterance: str, message_type: str, config: dict) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    message_id = str(uuid.uuid4())

    payload = {
        "service": "ambient-reminder",
        "message_id": message_id,
        "severity": 0.9,
        "message": utterance,
        "message_type": message_type,
        "external_id": endpoint_id,
        "occurred_at": now,
        "delivery_by": now,
        "utterance": utterance,
        "require_affirmation": False,
    }

    response = requests.post(config["CA_ENDPOINT"], json=payload, headers={
                             "Content-Type": "application/json",
                             "X-Robin-Key": config["CA_API_KEY"]})
    response.raise_for_status()

    print(f"CA {message_type} message sent successfully with message_id: {message_id} "
          f"-> {response.json()}")

    return True


def send_msg_to_ca(message: str, explanation: str, endpoint_id: str, config: dict) -> bool:

    try:
        # Send passive message first if explanation is provided
        if explanation and explanation.strip():
            _post_to_ca(endpoint_id, explanation, "passive", config)
        # Always send proactive message
        return _post_to_ca(endpoint_id, message, "proactive", config)

    except Exception as e:
        print(f"Failed to send CA message: {e}")
        return False


def _config():
    key = os.environ.get("ROBIN_API_KEY")
    if not key:
        env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        for line in open(env, encoding="utf-8"):
            if line.startswith("ROBIN_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return {"CA_ENDPOINT": os.environ.get("CA_ENDPOINT", DEFAULT_ENDPOINT),
            "CA_API_KEY": key}


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "The fridge is open."
    explanation = sys.argv[2] if len(sys.argv) > 2 else ""
    endpoint_id = os.environ.get("ENDPOINT_ID", "ambient-reminder-dev")
    sys.exit(0 if send_msg_to_ca(message, explanation, endpoint_id, _config()) else 1)
