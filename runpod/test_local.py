"""Run the handler locally without the RunPod runtime.

CPU fallback so this works on Macs and any non-GPU dev box. Useful for shaking
out the input/output contract before paying for cloud GPU minutes.

    python runpod/test_local.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
"""

from __future__ import annotations

import json
import sys

from handler import handler


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: test_local.py <url-or-file>")
        return 2
    event = {
        "input": {
            "url_or_file": sys.argv[1],
            "device": "cpu",
            "model": "tiny",
            "task": "transcribe",
        }
    }
    result = handler(event)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
