from __future__ import annotations

import http.server
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

from messy_napkins.benchmark import (
    VramSampler,
    aggregate_trials,
    estimate_token_count,
    evaluate_with_aislop,
    run_benchmark,
    run_prompt,
    run_prompt_http,
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

AISLOP_MULTI_SCORE_COMMAND = [
    sys.executable,
    "-c",
    "import json,sys; text=sys.stdin.read(); n=len(text); print(json.dumps({'correctness': n/10, 'style': n/20}))",
]

AISLOP_FLOAT_SCORE_COMMAND = [sys.executable, "-c", "print('0.42')"]
FAST_RUNNER_COMMAND = [sys.executable, "-c", "import sys; print('done:' + sys.argv[1])"]
SLOW_RUNNER_COMMAND = [sys.executable, "-c", "import time; time.sleep(2); print('too-late')"]


@contextmanager
def _fake_openai_server(response_json: dict):
    """Serve a canned OpenAI-compatible chat completion response on localhost."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (required override name)
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps(response_json).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        server_thread.join(timeout=2)
        server.server_close()


def _base_config(output_path: str, *, trials: int = 1) -> dict:
    return {
        "name": "test-run",
        "model": {
            "id": "tiny-local",
            "backend": "rocm",  # backward-compat flat backend string
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
        "output": {"path": output_path},
        "cases": [
            {
                "id": "case-1",
                "task": "code_gen",
                "prompt": "Return hello world",
                "trials": trials,
            }
        ],
    }


class BenchmarkRunnerTests(unittest.TestCase):
    def test_run_benchmark_full_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "logs" / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path)))

            results = run_benchmark(config)

            self.assertEqual(1, len(results))
            result = results[0]
            self.assertEqual("case-1", result["case_id"])
            self.assertEqual("trial", result["row_type"])
            self.assertEqual(0, result["trial_index"])
            self.assertGreater(result["tokens_per_second"], 0)
            self.assertGreater(result["output_tokens"], 0)
            self.assertEqual("estimated", result["output_token_source"])
            self.assertIn("vram_used_mb_peak", result)
            self.assertIn("vram_used_mb_avg", result)
            self.assertIsInstance(result["quality_scores"], dict)
            self.assertGreaterEqual(result["quality_scores"]["total"], 1)
            self.assertTrue(output_path.exists())

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(1, len(lines))
            persisted = json.loads(lines[0])
            self.assertEqual(result["case_id"], persisted["case_id"])
            self.assertEqual(result["generated_output"], persisted["generated_output"])

    def test_run_benchmark_backward_compat_backend_field(self) -> None:
        """Plain "backend" string in model config maps to engine.accelerator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path)))
            self.assertEqual("rocm", config.model.engine.accelerator)
            self.assertEqual("", config.model.engine.name)

    def test_run_benchmark_with_trials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "logs" / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path), trials=3))

            results = run_benchmark(config)

            # 3 trial rows + 1 aggregate row
            self.assertEqual(4, len(results))
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            agg_rows = [r for r in results if r["row_type"] == "aggregate"]
            self.assertEqual(3, len(trial_rows))
            self.assertEqual(1, len(agg_rows))

            # Trial indices are 0, 1, 2
            self.assertEqual([0, 1, 2], [r["trial_index"] for r in trial_rows])

            # Aggregate statistics
            agg = agg_rows[0]
            self.assertEqual("case-1", agg["case_id"])
            self.assertEqual(3, agg["trial_count"])
            self.assertEqual(3, agg["pass_count"])
            self.assertAlmostEqual(1.0, agg["pass_rate"])
            self.assertIsNotNone(agg["mean_ttft_seconds"])
            self.assertIsNotNone(agg["mean_tokens_per_second"])
            self.assertIn("total", agg["quality_scores_aggregate"])
            self.assertIsNotNone(agg["quality_scores_aggregate"]["total"]["mean"])

            # JSONL has 4 rows
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(4, len(lines))
            persisted_agg = json.loads(lines[3])
            self.assertEqual("aggregate", persisted_agg["row_type"])

    def test_run_benchmark_new_model_fields_in_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "model": {
                        "id": "phi-3-mini",
                        "engine": {
                            "name": "ollama",
                            "version": "0.1.34",
                            "accelerator": "cuda",
                        },
                        "hardware": {
                            "gpu": "RTX 4090",
                            "vram_gb": 24.0,
                            "cpu": "i9-13900K",
                            "os": "Ubuntu 24.04",
                        },
                        "quantization": "Q4_K_M",
                        "parameter_count": "3.8B",
                        "seed": 42,
                        "max_tokens": 256,
                        "context_size": 4096,
                        "parameters": {"temperature": 0.0},
                    },
                }
            )
            results = run_benchmark(config)
            row = results[0]
            self.assertEqual("ollama", row["engine_name"])
            self.assertEqual("0.1.34", row["engine_version"])
            self.assertEqual("cuda", row["engine_accelerator"])
            self.assertEqual("RTX 4090", row["hardware_gpu"])
            self.assertEqual(24.0, row["hardware_vram_gb"])
            self.assertEqual("Q4_K_M", row["model_quantization"])
            self.assertEqual("3.8B", row["model_parameter_count"])
            self.assertEqual(42, row["seed"])
            self.assertEqual(256, row["max_tokens"])

    def test_evaluate_with_aislop_accepts_float_stdout(self) -> None:
        scores = evaluate_with_aislop(
            command=AISLOP_FLOAT_SCORE_COMMAND,
            generated_output="ignored",
            timeout_seconds=10,
        )
        self.assertIsInstance(scores, dict)
        self.assertAlmostEqual(0.42, scores["total"])

    def test_evaluate_with_aislop_accepts_legacy_json_score(self) -> None:
        scores = evaluate_with_aislop(
            command=AISLOP_JSON_SCORE_COMMAND,
            generated_output="hello",
            timeout_seconds=10,
        )
        self.assertIn("total", scores)
        self.assertIsInstance(scores["total"], float)

    def test_evaluate_with_aislop_accepts_multi_dimensional_scores(self) -> None:
        scores = evaluate_with_aislop(
            command=AISLOP_MULTI_SCORE_COMMAND,
            generated_output="hello",
            timeout_seconds=10,
        )
        self.assertIn("correctness", scores)
        self.assertIn("style", scores)
        self.assertNotIn("total", scores)

    def test_benchmark_case_decode_tps_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path)))
            results = run_benchmark(config)
            row = results[0]
            # decode_tokens_per_second may be None (when ttft≈total) or a positive float
            decode_tps = row["decode_tokens_per_second"]
            self.assertTrue(decode_tps is None or decode_tps > 0)

    def test_aggregate_trials_statistics(self) -> None:
        fake_trials = [
            {
                "row_type": "trial",
                "trial_index": i,
                "benchmark_name": "test",
                "case_id": "c1",
                "task": "code_gen",
                "ttft_seconds": 0.1 * (i + 1),
                "total_seconds": 1.0 * (i + 1),
                "tokens_per_second": 100.0 / (i + 1),
                "decode_tokens_per_second": 110.0 / (i + 1),
                "quality_scores": {"total": float(i + 1)},
            }
            for i in range(3)
        ]
        agg = aggregate_trials("c1", "code_gen", fake_trials)
        self.assertEqual("aggregate", agg["row_type"])
        self.assertEqual(3, agg["trial_count"])
        self.assertEqual(3, agg["pass_count"])
        self.assertAlmostEqual(1.0, agg["pass_rate"])
        self.assertIsNotNone(agg["mean_ttft_seconds"])
        self.assertIsNotNone(agg["stddev_ttft_seconds"])
        self.assertIn("total", agg["quality_scores_aggregate"])
        self.assertAlmostEqual(2.0, agg["quality_scores_aggregate"]["total"]["mean"])

    def test_estimate_token_count_edge_cases(self) -> None:
        self.assertEqual(1, estimate_token_count(""))
        self.assertEqual(1, estimate_token_count("   \n\t"))
        self.assertEqual(3, estimate_token_count("one two three"))
        self.assertEqual(2, estimate_token_count("word1  \n\tword2"))
        self.assertEqual(2, estimate_token_count("🚀-launch\nnaïve,café"))

    def test_run_prompt_timeout_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            run_prompt(command=SLOW_RUNNER_COMMAND, prompt="ignored", timeout_seconds=0.5)

    def test_run_prompt_completes_within_timeout(self) -> None:
        output, ttft_seconds, total_seconds = run_prompt(
            command=FAST_RUNNER_COMMAND,
            prompt="ok",
            timeout_seconds=5,
        )
        self.assertEqual("done:ok", output)
        self.assertGreater(total_seconds, 0)
        self.assertGreaterEqual(total_seconds, ttft_seconds)

    def test_run_prompt_http_uses_api_reported_token_count(self) -> None:
        response = {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"completion_tokens": 7},
        }
        with _fake_openai_server(response) as url:
            output, ttft_seconds, total_seconds, api_token_count = run_prompt_http(
                url=url,
                model_id="test-model",
                prompt="hi",
                parameters={},
                seed=None,
                max_tokens=None,
                timeout_seconds=5,
            )
        self.assertEqual("hello world", output)
        self.assertEqual(7, api_token_count)
        self.assertGreaterEqual(total_seconds, 0)
        self.assertEqual(ttft_seconds, total_seconds)

    def test_run_prompt_http_falls_back_when_usage_missing(self) -> None:
        response = {"choices": [{"message": {"content": "no usage field here"}}]}
        with _fake_openai_server(response) as url:
            _, _, _, api_token_count = run_prompt_http(
                url=url,
                model_id="test-model",
                prompt="hi",
                parameters={},
                seed=None,
                max_tokens=None,
                timeout_seconds=5,
            )
        self.assertIsNone(api_token_count)

    def test_benchmark_case_http_runner_uses_api_token_source(self) -> None:
        response = {
            "choices": [{"message": {"content": "hello from the api"}}],
            "usage": {"completion_tokens": 4},
        }
        with _fake_openai_server(response) as url, tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "runner": {"type": "http", "url": url, "timeout_seconds": 10},
                }
            )
            results = run_benchmark(config)
            row = results[0]
            self.assertEqual("api", row["output_token_source"])
            self.assertEqual(4, row["output_tokens"])

    def test_vram_sampler_computes_peak_and_average(self) -> None:
        values = iter([100.0, 200.0, 150.0, 150.0, 150.0, 150.0])
        sampler = VramSampler(sample_fn=lambda: next(values, 150.0), interval_seconds=0.01)
        self.assertTrue(sampler.available)
        sampler.start()
        time.sleep(0.05)
        peak, avg = sampler.stop()
        self.assertIsNotNone(peak)
        self.assertIsNotNone(avg)
        assert peak is not None and avg is not None
        self.assertGreaterEqual(peak, avg)
        self.assertEqual(200.0, peak)

    def test_vram_sampler_unavailable_reports_none(self) -> None:
        sampler = VramSampler(sample_fn=None, interval_seconds=0.01)
        # Force "unavailable" regardless of what's actually on the test host's PATH.
        sampler._sample_fn = None  # noqa: SLF001 (test-only override)
        self.assertFalse(sampler.available)
        sampler.start()
        peak, avg = sampler.stop()
        self.assertIsNone(peak)
        self.assertIsNone(avg)


if __name__ == "__main__":
    unittest.main()
