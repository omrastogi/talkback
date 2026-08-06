"""Interactively test Robin's full conversation logic (this package) -- intent routing,
the delete-confirmation gate, weather/schedule/capabilities replies, and the default LLM
turn. No Flask app, no database, no AWS Bedrock.

This script is a dev convenience, not part of the portable API -- everything else in this
package only needs OPENAI_API_KEY (see llm.py). The talkback wiring below is just how this
particular machine happens to have a usable key; drop it and set OPENAI_API_KEY yourself if
running this elsewhere.

Run as a module from backend/ (one level above this package):
    python -m robin_conversation.test_conversation                  # OpenAI gpt-4o-mini (default)
    python -m robin_conversation.test_conversation --backend parcs   # PARCS gemma gateway
"""
import argparse
import logging
import os
import sys

TALKBACK_DIR = "/home/omrastogi/Project/talkback"


def _configure_from_talkback(backend: str) -> str:
    sys.path.insert(0, TALKBACK_DIR)
    import config as talkback_config  # talkback's own .env loader + backend resolver

    base_url, api_key, model = talkback_config.resolve_backend(backend)
    if not api_key:
        sys.exit(f"No API key configured for backend '{backend}' in {TALKBACK_DIR}/.env")

    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["CHAT_MODEL_ID"] = model
    os.environ["INTENT_MODEL_ID"] = model
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Chat with Robin's standalone conversation package directly."
    )
    parser.add_argument("--backend", choices=["openai", "parcs"], default="openai")
    args = parser.parse_args()

    model = _configure_from_talkback(args.backend)
    logging.basicConfig(level=logging.WARNING)

    # Imported only after env vars are set: .llm reads them at import time.
    from . import process_turn

    history = []
    user_id = "test-user"
    print(f"Talking to Robin via {args.backend} ({model}). Type 'quit' to exit.\n")
    while True:
        try:
            content = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not content:
            continue
        if content.lower() in {"quit", "exit"}:
            break

        result = process_turn(history, content, user_id=user_id)
        print(f"[intent={result['intent']}] robin> {result['reply']}\n")
        if result["should_end_session"]:
            break


if __name__ == "__main__":
    main()
