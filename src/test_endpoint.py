"""
test_endpoint.py

Purpose:
    Standalone sanity-check client for the deployed Modal /generate endpoint.
    Auth is enforced (shared-secret header) -- pass --secret with the value
    of SHARED_SECRET_TOKEN you created via `modal secret create`.

Usage:
    python test_endpoint.py --url <endpoint_url> --secret <shared_secret> --prompt "hello"
"""

import argparse
import json
import sys
import time

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Modal Llama 3.2 3B endpoint")
    parser.add_argument("--url", required=True, help="Full /generate endpoint URL")
    parser.add_argument("--secret", required=True, help="Shared secret (X-Auth-Token value)")
    parser.add_argument("--prompt", default="Hello, who are you?", help="Prompt to send")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": args.secret,
    }

    print(f"POST {args.url}")
    print(f"Payload: {json.dumps(payload)}")

    start = time.time()
    try:
        resp = requests.post(args.url, json=payload, headers=headers, timeout=120)
    except requests.exceptions.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    elapsed = round(time.time() - start, 2)

    print(f"Status: {resp.status_code}  (round-trip: {elapsed}s)")
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)

    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())