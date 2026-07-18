"""Create a local context-window boundary benchmark from the local config."""

from __future__ import annotations

import copy
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "configs" / "local" / "qwen3-14b-gguf.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="configs/local/qwen3-context-boundary.json",
        help="Path for the generated local config",
    )
    parser.add_argument(
        "--results",
        default="logs/qwen3-context-boundary-results.jsonl",
        help="JSONL output path recorded in the generated config",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=int,
        default=[16000, 32000, 39000, 42000],
        help="Approximate prompt token targets",
    )
    args = parser.parse_args()

    with SOURCE.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["name"] = "qwen3-14b-context-boundary"
    config["model"]["max_tokens"] = 8
    config["model"]["context_size"] = 40960
    config["runner"]["timeout_seconds"] = 180
    config["output"]["path"] = args.results

    cases = []
    for target_tokens in args.targets:
        prompt = (
            "Reply with exactly BOUNDARY_OK and nothing else. "
            + ("context " * target_tokens)
        )
        case = {
            "id": f"context-boundary-{target_tokens}",
            "task": "instruction_following",
            "prompt": prompt,
            "expected_answer": "BOUNDARY_OK",
            "pass_condition": "exact_match",
            "trials": 1,
            "warmup_trials": 0,
        }
        cases.append(case)

    config["cases"] = [copy.deepcopy(case) for case in cases]
    target = ROOT / args.output
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {target}")
    print("Probe sizes are approximate; use API-reported prompt_tokens as evidence.")


if __name__ == "__main__":
    main()