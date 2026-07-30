"""See LLM streaming with your eyes: tokens print live as they arrive,
and each finished sentence is flushed to the (fake) TTS queue with a marker.

    python stream_demo.py --text "Tell me about the ocean in 3 sentences."
"""
import argparse
import queue
import re
import sys
import time

import config

tts_queue = queue.Queue()  # real cascade hands this to the TTS stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Tell me about the ocean in three short sentences.")
    ap.add_argument("--backend", choices=["openai", "parcs"], default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client, model = config.make_client(args.backend, args.model)
    messages = [
        {"role": "system", "content": config.SYS_PROMPT},
        {"role": "user", "content": args.text},
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    buf, full = "", ""
    t0 = time.time()
    first = None
    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""   # can be None on first/last chunk
        if not piece:
            continue
        if first is None:
            first = time.time() - t0
        sys.stdout.write(piece)                        # live token print — this is the streaming
        sys.stdout.flush()
        buf += piece
        full += piece
        while (m := re.search(r"[.!?]", buf)):          # flush finished sentences to TTS
            sentence, buf = buf[:m.end()], buf[m.end():]
            tts_queue.put(sentence)
            sys.stdout.write(f"\n   → [TTS] {sentence.strip()}\n")  # marker so you see the flush
            sys.stdout.flush()
    if buf.strip():
        tts_queue.put(buf)
        sys.stdout.write(f"\n   → [TTS] {buf.strip()}\n")

    messages.append({"role": "assistant", "content": full})  # keep history
    total = time.time() - t0
    print(f"\nTIMING first_token={first:.2f}s total={total:.2f}s sentences={tts_queue.qsize()}")


if __name__ == "__main__":
    main()
