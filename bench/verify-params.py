#!/usr/bin/env python3
"""Read arm-defining parameters back from a running llama-server and emit a
receipt. Implements P1 of bench/PROTOCOL-RULES.md for the server instrument.

The probe uses the arm's OWN request body, so a per-request override that the
build silently ignores shows up as a mismatch here rather than as a clean,
tight-spread, wrong number later. This is the check that decode-race arm B
needed: "speculative.n_max": 0 was accepted and ignored, draft_n unchanged.

  ./bench/verify-params.py --arm B --body arm-b.json \
      --expect timings.draft_n=0 --expect props.default_generation_settings.n_ctx=131072 \
      --out .work/receipts/arm-b.json

Exit non-zero on any mismatch, missing key, or unreachable server. No receipt
file is written on failure: a void arm must leave no artifact that could later
be mistaken for a passing one.
"""

import argparse, json, sys, time, urllib.request, urllib.error


def fetch(url, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def dig(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8083")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--body", help="JSON file: the arm's exact /completion request body")
    ap.add_argument("--expect", action="append", default=[],
                    help="dotted.path=value, checked against {props,timings}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    try:
        props = fetch(a.url + "/props")
    except (urllib.error.URLError, OSError) as e:
        print(f"VOID arm {a.arm}: server unreachable at {a.url} ({e})")
        return 1

    timings = {}
    if a.body:
        body = json.load(open(a.body))
        # A probe long enough for the draft path to actually engage; a
        # 1-token probe reports draft_n 0 for every arm and proves nothing.
        probe = dict(body)
        probe.update({"n_predict": max(16, int(body.get("n_predict", 16)) // 4),
                      "cache_prompt": False})
        try:
            timings = fetch(a.url + "/completion", probe).get("timings", {})
        except (urllib.error.URLError, OSError) as e:
            print(f"VOID arm {a.arm}: probe request failed ({e})")
            return 2
        if not timings:
            print(f"VOID arm {a.arm}: probe returned no timings block")
            return 2

    doc = {"props": props, "timings": timings}
    failures = []
    checked = {}
    for spec in a.expect:
        if "=" not in spec:
            failures.append(f"malformed --expect {spec!r}")
            continue
        path, want = spec.split("=", 1)
        got, present = dig(doc, path)
        if not present:
            failures.append(f"{path}: NOT REPORTED by server (unverifiable "
                            f"per-request knob -> define this arm by restart)")
            continue
        checked[path] = got
        if str(got) != want:
            failures.append(f"{path}: expected {want}, server reports {got}")

    if failures:
        print(f"VOID arm {a.arm}: parameter verification FAILED")
        for f in failures:
            print("  -", f)
        return 3

    receipt = {"arm": a.arm, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "url": a.url, "checked": checked, "timings": timings,
               "props": props}
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(receipt, open(a.out, "w"), indent=2)
    print(f"arm {a.arm}: {len(checked)} parameter(s) verified -> {a.out}")
    for k, v in checked.items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
