"""The one case an ordinary client cannot provoke: a send that dies PART WAY through.

Synthesis is ~0.04s a sentence and WebSocket writes are buffered, so a whole utterance leaves
the server in well under a second -- far too fast to abort from outside. This drives a RAW
socket that completes the handshake and then never reads: the kernel receive buffer fills, the
server blocks mid-write, and an RST makes that in-flight write raise. The endpoint must report
`queued`, never `spoken`, and must not lose the message.

Plain TCP only, so it runs against a LOCAL server -- not through a TLS tunnel:

    python test/test_proactive_midsend.py            # 127.0.0.1:8000 by default
    PROACTIVE_PORT=9001 python test/test_proactive_midsend.py
"""
import base64
import json
import os
import socket
import struct
import sys
import threading
import time

import requests

from proactive_common import HDRS, KEY, Results, payload

R = Results()
HOST = os.environ.get("PROACTIVE_HOST", "127.0.0.1")
PORT = int(os.environ.get("PROACTIVE_PORT", "8000"))
EXT = os.environ.get("PROACTIVE_EXT", "test-endpoint-midsend")
# Long enough that the audio cannot fit in any socket buffer, so the server is still writing
# when the reset lands.
LONG = " ".join(f"This is sentence {i} of a very long proactive message." for i in range(1, 40))


def masked_text_frame(text):
    """A client-to-server WebSocket text frame (client frames must be masked)."""
    data = text.encode()
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    n = len(data)
    header = b"\x81" + (bytes([0x80 | n]) if n < 126 else b"\xfe" + struct.pack(">H", n))
    return header + mask + masked


def main():
    print(f"\ntarget {HOST}:{PORT}  external_id={EXT}")
    s = socket.create_connection((HOST, PORT))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)        # a tiny window fills fast
    s.sendall((f"GET /ws-stream HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {base64.b64encode(os.urandom(16)).decode()}\r\n"
               f"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: {KEY}\r\n\r\n").encode())
    status = s.recv(200).split(b"\r\n")[0].decode()
    R.check("handshake accepted", "101" in status, status)
    s.sendall(masked_text_frame(json.dumps({"type": "start", "sampleRate": 16000,
                                            "external_id": EXT})))
    time.sleep(0.5)                                                # let the server bind external_id

    body, out = payload(LONG, external_id=EXT), {}
    t = threading.Thread(target=lambda: out.update(
        requests.post(f"http://{HOST}:{PORT}/proactive", json=body, headers=HDRS, timeout=120).json()))
    t.start()
    time.sleep(3.0)                                                # the socket is full by now
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    s.close()                                                      # RST: the in-flight write raises
    print("   socket reset while the server was mid-utterance")
    t.join()
    R.check("queued, not spoken", out.get("status") == "queued", out)
    R.check("the caller gets its message_id back", out.get("message_id") == body["message_id"], out)
    print("   (server.log should show: proactive send failed ... -- queueing)")
    return R.report()


sys.exit(main())
