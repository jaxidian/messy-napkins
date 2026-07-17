from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from messy_napkins.benchmark import (
    estimate_token_count,
    evaluate_with_aislop,
    run_benchmark,
    run_prompt,
)
from messy_napkins.config import BenchmarkConfig

RUNNER_COMMAND = [
    sys.executable,
    "-c",
    "import sys; print('generated:' + sys.argv[1])",
]

AISLOP_JSON_SCORE_COMMAND = [
    sys.executable,
    "-c",
    "import json,sys; text=sys.stdin.read(); print(json.dumps({'score': len(text)}))",
]

AISLOP_FLOAT_SCORE_COMMAND = [sys.executable, "-c", "print('0.42')"]
SLOW_RUNNER_COMMAND = [sys.executable, "-c", "import time; time.sleep(1); print('too-late')"]


class BenchmarkRunnerTests(unittest.TestCase):
    def test_run_benchmark_full_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "logs" / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    "name": "test-run",
                    "model": {
                        "id": "tiny-local",
                        "backend": "rocm",
                        "context_size": 8192,
                        "parameters": {"temperature": 0.1},
                    },
                    "runner": {
                        "command": RUNNER_COMMAND,
                        "timeout_seconds": 10,
                    },
                    "aislop": {
                        "command": AISLOP_JSON_SCORE_COMMAND,
                        "timeout_seconds": 10,
                    },
                    "output": {"path": str(output_path)},
                    "cases": [
                        {
                            "id": "case-1",
                            "task": "code_gen",
                            "prompt": "Return hello world",
                        }
                    ],
                }
            )

            results = run_benchmark(config)

            self.assertEqual(1, len(results))
            result = results[0]
            self.assertEqual("case-1", result["case_id"])
            self.assertGreater(result["tokens_per_second"], 0)
            self.assertGreaterEqual(result["quality_score"], 1)
            self.assertTrue(output_path.exists())

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(1, len(lines))
            persisted = json.loads(lines[0])
            self.assertEqual(result["case_id"], persisted["case_id"])
            self.assertEqual(result["generated_output"], persisted["generated_output"])

    def test_evaluate_with_aislop_accepts_float_stdout(self) -> None:
        score = evaluate_with_aislop(
            command=AISLOP_FLOAT_SCORE_COMMAND,
            generated_output="ignored",
            timeout_seconds=10,
        )
        self.assertAlmostEqual(0.42, score)

    def test_estimate_token_count_edge_cases(self) -> None:
        self.assertEqual(1, estimate_token_count(""))
        self.assertEqual(1, estimate_token_count("   \n\t"))
        self.assertEqual(3, estimate_token_count("one two three"))
        self.assertEqual(2, estimate_token_count("word1  \n\tword2"))
        self.assertEqual(2, estimate_token_count("🚀-launch\nnaïve,café"))

    def test_run_prompt_timeout_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            run_prompt(command=SLOW_RUNNER_COMMAND, prompt="ignored", timeout_seconds=0.01)


if __name__ == "__main__":
    unittest.main()
