"""TTS stage: Kokoro-82M. Writes 24 kHz wav."""
import argparse
import os
import time

import numpy as np
import soundfile as sf


def load_pipe():
    from kokoro import KPipeline
    return KPipeline(lang_code="a")


def synth(pipe, text):
    chunks = []
    for gs, ps, audio in pipe(text, voice="af_heart"):
        chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--in", dest="infile")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    text = args.text if args.text is not None else open(args.infile, encoding="utf-8").read().strip()

    t0 = time.time()
    pipe = load_pipe()
    load = time.time() - t0

    t1 = time.time()
    audio = synth(pipe, text)
    infer = time.time() - t1

    sf.write(args.out, audio, 24000)   # Kokoro is 24 kHz
    print(f"TIMING stage=tts load={load:.2f} infer={infer:.2f}")


if __name__ == "__main__":
    main()
