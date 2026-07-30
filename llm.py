"""LLM stage: OpenAI-compatible chat completion (backend: openai, or the parcs gemma gateway).

Runs off-GPU, so no VRAM tiering here — the cascade still runs it as an isolated
subprocess to keep the stage contract uniform.
"""
import argparse
import os
import time

import config


def reply(client, text, model):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": config.SYS_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=200,
        temperature=0.6,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--in", dest="infile")
    ap.add_argument("--out", default="out/reply.txt")
    ap.add_argument("--backend", choices=["openai", "parcs"], default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    user_text = args.text if args.text is not None else open(args.infile, encoding="utf-8").read().strip()

    t0 = time.time()
    client, model = config.make_client(args.backend, args.model)
    load = time.time() - t0

    t1 = time.time()
    out = reply(client, user_text, model)
    infer = time.time() - t1

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    backend = args.backend or os.environ.get("LLM_BACKEND", "parcs")
    print(f"Model: {model} ({backend})")
    print(out)
    print(f"TIMING stage=llm load={load:.2f} infer={infer:.2f}")


if __name__ == "__main__":
    main()
