"""Gate-4 self-test: drive /ws without a browser. Send out/hello.wav bytes, assert we get
a transcript text frame, >=1 binary audio frame, and a done. ffmpeg decodes the wav through
the same path as browser audio, so this exercises the full pipeline."""
import asyncio
import json

import websockets


async def main():
    data = open("out/hello.wav", "rb").read()
    async with websockets.connect("ws://localhost:8000/ws", max_size=None) as ws:
        await ws.send(data)
        got_transcript = got_audio = got_done = False
        while not got_done:
            msg = await asyncio.wait_for(ws.recv(), timeout=90)
            if isinstance(msg, (bytes, bytearray)):
                got_audio = True
            else:
                m = json.loads(msg)
                if m["type"] == "transcript":
                    got_transcript = True
                    print("transcript:", m["text"])
                elif m["type"] == "reply":
                    print("reply:", m["text"])
                elif m["type"] == "done":
                    got_done = True
        assert got_transcript, "no transcript frame"
        assert got_audio, "no audio frame"
        assert got_done, "no done frame"
        print("PASS: transcript + audio + done")


if __name__ == "__main__":
    asyncio.run(main())
