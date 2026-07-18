"""Bridge benchmark evaluator payloads to the aislop source scanner."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _code_to_scan(payload: dict[str, object]) -> str | None:
    if payload.get("task") != "code_gen":
        return None

    generated_output = str(payload.get("generated_output") or "")
    match = PYTHON_BLOCK_RE.search(generated_output)
    return match.group(1) if match else generated_output


def _aislop_command() -> list[str]:
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise RuntimeError(
            "npx was not found on PATH. Install Node.js 20+ to run the aislop evaluator."
        )
    return [npx, "--yes", "aislop@latest", "scan"]


def main() -> int:
    payload = json.load(sys.stdin)
    source = _code_to_scan(payload)
    if source is None:
        # aislop analyzes source code, so it cannot score documentation or
        # exact-answer cases as a code-quality metric.
        print(json.dumps({}))
        return 0

    with tempfile.TemporaryDirectory(prefix="messy-napkins-aislop-") as temp_dir:
        source_path = Path(temp_dir) / "generated.py"
        source_path.write_text(source, encoding="utf-8")
        subprocess.run(
            ["git", "init", "--quiet"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "add", "generated.py"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        completed = subprocess.run(
            [*_aislop_command(), temp_dir, "--json"],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
            env={**os.environ, "CI": "true"},
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(
                f"aislop failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )

        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"aislop did not return JSON: {completed.stdout.strip() or completed.stderr.strip()}"
            ) from exc

        score = result.get("score")
        if score is None or result.get("scoreable") is False:
            print(json.dumps({}))
        else:
            print(json.dumps({"aislop_score": float(score)}))


if __name__ == "__main__":
    raise SystemExit(main())
