from __future__ import annotations

import http.server
import json
import statistics
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
    build_run_manifest,
    estimate_token_count,
    evaluate_with_aislop,
    run_benchmark,
    run_prompt,
    run_prompt_http,
    run_prompt_http_streaming,
)
from messy_napkins.config import BenchmarkConfig

# ---------------------------------------------------------------------------
# Shared test commands
# ---------------------------------------------------------------------------

RUNNER_COMMAND = [
    sys.executable,
    "-c",
    "import sys; print('generated:' + sys.argv[1])",
]

# Updated: evaluators now receive a JSON payload on stdin rather than raw text.
AISLOP_JSON_SCORE_COMMAND = [
    sys.executable,
    "-c",
    "import json,sys; p=json.loads(sys.stdin.read()); print(json.dumps({'score': len(p['generated_output'])}))",
]

AISLOP_MULTI_SCORE_COMMAND = [
    sys.executable,
    "-c",
    (
        "import json,sys; p=json.loads(sys.stdin.read()); n=len(p['generated_output']); "
        "print(json.dumps({'correctness': n/10, 'style': n/20}))"
    ),
]

# Does not read stdin — valid for evaluators that ignore context.
AISLOP_FLOAT_SCORE_COMMAND = [sys.executable, "-c", "print('0.42')"]

FAST_RUNNER_COMMAND = [sys.executable, "-c", "import sys; print('done:' + sys.argv[1])"]
SLOW_RUNNER_COMMAND = [sys.executable, "-c", "import time; time.sleep(2); print('too-late')"]
FAILING_RUNNER_COMMAND = [sys.executable, "-c", "import sys; sys.exit(1)"]
EMPTY_OUTPUT_RUNNER_COMMAND = [sys.executable, "-c", "import sys; _ = sys.argv[1]"]


# ---------------------------------------------------------------------------
# Test server helpers
# ---------------------------------------------------------------------------

@contextmanager
def _fake_openai_server(response_json: dict):
    """Serve a canned non-streaming OpenAI-compatible chat completion response."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
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


@contextmanager
def _fake_streaming_openai_server(
    tokens: list[str] | None = None,
    first_token_delay_seconds: float = 0.05,
    completion_tokens: int = 2,
    prompt_tokens: int = 3,
):
    """Serve a chunked SSE streaming OpenAI-compatible response.

    Delays ``first_token_delay_seconds`` before the first content token so
    tests can verify that ``ttft_seconds < total_seconds``.
    """
    if tokens is None:
        tokens = ["hello", " world"]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write_sse_chunk(data: str) -> None:
                line = f"data: {data}\n\n"
                encoded = line.encode("utf-8")
                size_prefix = f"{len(encoded):x}\r\n".encode("ascii")
                self.wfile.write(size_prefix + encoded + b"\r\n")
                self.wfile.flush()

            time.sleep(first_token_delay_seconds)
            for token in tokens:
                write_sse_chunk(
                    json.dumps({"choices": [{"delta": {"content": token}, "finish_reason": None}]})
                )
                time.sleep(0.01)
            # Final chunk with usage
            write_sse_chunk(
                json.dumps({
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                    },
                })
            )
            write_sse_chunk("[DONE]")
            # Terminate chunked encoding
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

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


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _base_config(output_path: str, *, trials: int = 1, warmup_trials: int = 0) -> dict:
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
                "warmup_trials": warmup_trials,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class BenchmarkRunnerTests(unittest.TestCase):

    # --- Full workflow ---

    def test_run_benchmark_full_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "logs" / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path)))

            results = run_benchmark(config)

            # 1 manifest + 1 trial row
            self.assertEqual(2, len(results))
            manifest = results[0]
            result = results[1]

            # Manifest
            self.assertEqual("run_manifest", manifest["row_type"])
            self.assertIn("run_id", manifest)
            self.assertIn("config_hash", manifest)
            self.assertIn("python_version", manifest)
            self.assertEqual("test-run", manifest["benchmark_name"])

            # Trial row
            self.assertEqual("case-1", result["case_id"])
            self.assertEqual("trial", result["row_type"])
            self.assertEqual(0, result["trial_index"])
            self.assertFalse(result["error"])
            # Run ID is consistent
            self.assertEqual(manifest["run_id"], result["run_id"])
            # Token metrics
            self.assertGreater(result["output_tokens"], 0)
            self.assertEqual("estimated", result["output_token_source"])
            self.assertIsNotNone(result["tokens_per_second"])
            self.assertGreater(result["tokens_per_second"], 0)
            # Effective command present; effective_request absent for subprocess
            self.assertIsNotNone(result["effective_command"])
            self.assertIsNone(result["effective_request"])
            # VRAM telemetry fields (new baseline + delta fields)
            self.assertIn("vram_peak_mb", result)
            self.assertIn("vram_baseline_mb", result)
            self.assertIn("vram_peak_delta_mb", result)
            self.assertIn("vram_avg_delta_mb", result)
            # Quality
            self.assertIsInstance(result["quality_scores"], dict)
            self.assertGreaterEqual(result["quality_scores"]["total"], 1)

            # JSONL: 2 rows
            self.assertTrue(output_path.exists())
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(2, len(lines))
            persisted = json.loads(lines[1])
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

            # 1 manifest + 3 trial rows + 1 aggregate row
            self.assertEqual(5, len(results))
            manifest_rows = [r for r in results if r["row_type"] == "run_manifest"]
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            agg_rows = [r for r in results if r["row_type"] == "aggregate"]
            self.assertEqual(1, len(manifest_rows))
            self.assertEqual(3, len(trial_rows))
            self.assertEqual(1, len(agg_rows))

            # Trial indices are 0, 1, 2
            self.assertEqual([0, 1, 2], [r["trial_index"] for r in trial_rows])

            # Aggregate statistics (renamed from pass_rate)
            agg = agg_rows[0]
            self.assertEqual("case-1", agg["case_id"])
            self.assertEqual(3, agg["trial_count"])
            self.assertEqual(3, agg["execution_success_count"])
            self.assertAlmostEqual(1.0, agg["execution_success_rate"])
            # Timing stats
            self.assertIsNotNone(agg["mean_ttft_seconds"])
            self.assertIsNotNone(agg["median_ttft_seconds"])
            self.assertIsNotNone(agg["p95_ttft_seconds"])
            self.assertIsNotNone(agg["min_ttft_seconds"])
            self.assertIsNotNone(agg["max_ttft_seconds"])
            self.assertIsNotNone(agg["mean_tokens_per_second"])
            # Quality aggregate
            self.assertIn("total", agg["quality_scores_aggregate"])
            self.assertIsNotNone(agg["quality_scores_aggregate"]["total"]["mean"])
            self.assertIn("median", agg["quality_scores_aggregate"]["total"])

            # JSONL has 5 rows
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(5, len(lines))
            persisted_agg = json.loads(lines[4])
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
            # results[0] is the manifest; results[1] is the trial row
            row = results[1]
            self.assertEqual("ollama", row["engine_name"])
            self.assertEqual("0.1.34", row["engine_version"])
            self.assertEqual("cuda", row["engine_accelerator"])
            self.assertEqual("RTX 4090", row["hardware_gpu"])
            self.assertEqual(24.0, row["hardware_vram_gb"])
            self.assertEqual("Q4_K_M", row["model_quantization"])
            self.assertEqual("3.8B", row["model_parameter_count"])
            self.assertEqual(42, row["seed"])
            self.assertEqual(256, row["max_tokens"])

    # --- Trial failure handling ---

    def test_run_benchmark_records_failure_row_on_inference_error(self) -> None:
        """A failed inference call produces a failure row; the benchmark does not abort."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "runner": {"command": FAILING_RUNNER_COMMAND, "timeout_seconds": 5},
                }
            )
            results = run_benchmark(config)

            trial_rows = [r for r in results if r["row_type"] == "trial"]
            self.assertEqual(1, len(trial_rows))
            row = trial_rows[0]
            self.assertTrue(row["error"])
            self.assertEqual("inference", row["stage"])
            self.assertIsNotNone(row["error_message"])
            self.assertIsNone(row["tokens_per_second"])
            self.assertIsNone(row["generated_output"])

    def test_aggregate_trials_with_one_failure_gives_correct_execution_success_rate(self) -> None:
        """Acceptance criterion: timeout on 1 of 3 trials → execution_success_rate == 2/3."""
        trials = [
            {
                "row_type": "trial", "trial_index": 0, "benchmark_name": "t",
                "case_id": "c1", "task": "code_gen", "run_id": "x",
                "error": False,
                "ttft_seconds": 0.1, "total_seconds": 1.0,
                "tokens_per_second": 100.0, "decode_tokens_per_second": 110.0,
                "quality_scores": {"total": 0.9}, "task_passed": None,
            },
            {
                "row_type": "trial", "trial_index": 1, "benchmark_name": "t",
                "case_id": "c1", "task": "code_gen", "run_id": "x",
                "error": True, "error_type": "RuntimeError",
                "error_message": "Prompt command timed out after 5 seconds.",
                "timed_out": True, "stage": "inference",
                "ttft_seconds": None, "total_seconds": None,
                "tokens_per_second": None, "decode_tokens_per_second": None,
                "quality_scores": None, "task_passed": None,
            },
            {
                "row_type": "trial", "trial_index": 2, "benchmark_name": "t",
                "case_id": "c1", "task": "code_gen", "run_id": "x",
                "error": False,
                "ttft_seconds": 0.2, "total_seconds": 1.5,
                "tokens_per_second": 80.0, "decode_tokens_per_second": 85.0,
                "quality_scores": {"total": 0.7}, "task_passed": None,
            },
        ]
        agg = aggregate_trials("c1", "code_gen", trials)

        self.assertEqual(3, agg["trial_count"])
        self.assertEqual(2, agg["execution_success_count"])
        self.assertAlmostEqual(2 / 3, agg["execution_success_rate"])
        # Mean TTFT computed from the 2 successful trials only
        self.assertAlmostEqual(statistics.mean([0.1, 0.2]), agg["mean_ttft_seconds"])

    # --- Token metrics ---

    def test_estimate_token_count_edge_cases(self) -> None:
        # Empty / whitespace returns 0 (not 1) so downstream can detect "no output"
        self.assertEqual(0, estimate_token_count(""))
        self.assertEqual(0, estimate_token_count("   \n\t"))
        self.assertEqual(3, estimate_token_count("one two three"))
        self.assertEqual(2, estimate_token_count("word1  \n\tword2"))
        self.assertEqual(2, estimate_token_count("🚀-launch\nnaïve,café"))

    def test_empty_output_gives_null_tps(self) -> None:
        """Empty completion must not produce positive TPS; token source is 'unavailable'."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "runner": {"command": EMPTY_OUTPUT_RUNNER_COMMAND, "timeout_seconds": 10},
                }
            )
            results = run_benchmark(config)
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            self.assertEqual(1, len(trial_rows))
            row = trial_rows[0]
            self.assertFalse(row["error"])
            self.assertEqual(0, row["output_tokens"])
            self.assertEqual("unavailable", row["output_token_source"])
            self.assertIsNone(row["tokens_per_second"])
            self.assertIsNone(row["decode_tokens_per_second"])

    # --- Streaming HTTP inference ---

    def test_run_prompt_http_streaming_real_ttft(self) -> None:
        """TTFT reflects first content token arrival, not full response time."""
        with _fake_streaming_openai_server(
            first_token_delay_seconds=0.05,
            completion_tokens=2,
            prompt_tokens=3,
        ) as url:
            output, ttft_seconds, total_seconds, comp_tokens, prmpt_tokens = run_prompt_http_streaming(
                url=url,
                model_id="test-model",
                prompt="hi",
                parameters={},
                seed=None,
                max_tokens=None,
                timeout_seconds=10,
            )
        self.assertEqual("hello world", output)
        # Real TTFT must be strictly less than total (first token came before last)
        self.assertLess(ttft_seconds, total_seconds)
        self.assertGreater(ttft_seconds, 0)
        # decode TPS is based on post-first-token time
        decode_time = total_seconds - ttft_seconds
        self.assertGreater(decode_time, 0)
        # Usage from final chunk
        self.assertEqual(2, comp_tokens)
        self.assertEqual(3, prmpt_tokens)

    def test_benchmark_case_http_runner_uses_streaming_and_api_token_source(self) -> None:
        with _fake_streaming_openai_server(completion_tokens=4) as url, \
                tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "runner": {"type": "http", "url": url, "timeout_seconds": 10},
                }
            )
            results = run_benchmark(config)
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            row = trial_rows[0]
            self.assertEqual("api", row["output_token_source"])
            self.assertEqual(4, row["output_tokens"])
            # effective_request present; effective_command absent
            self.assertIsNotNone(row["effective_request"])
            self.assertIsNone(row["effective_command"])
            self.assertIn("stream", row["effective_request"])

    # --- Effective config ---

    def test_trial_row_includes_effective_command_for_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path)))
            results = run_benchmark(config)
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            row = trial_rows[0]
            self.assertIsNotNone(row["effective_command"])
            # Prompt is the last element of the effective command
            self.assertEqual("Return hello world", row["effective_command"][-1])
            self.assertIsNone(row["effective_request"])

    def test_trial_row_includes_effective_request_for_http(self) -> None:
        """effective_request for HTTP runner is the actual payload sent (provably executed config)."""
        response = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 2},
        }
        with _fake_openai_server(response) as url, tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "model": {
                        "id": "phi",
                        "context_size": 4096,
                        "parameters": {"temperature": 0.5},
                        "seed": 7,
                        "max_tokens": 128,
                    },
                    "runner": {"type": "http", "url": url, "timeout_seconds": 10},
                }
            )
            results = run_benchmark(config)
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            row = trial_rows[0]
            req = row["effective_request"]
            self.assertIsNotNone(req)
            self.assertEqual("phi", req["model"])
            self.assertEqual(0.5, req["temperature"])
            self.assertEqual(7, req["seed"])
            self.assertEqual(128, req["max_tokens"])

    # --- Evaluator context ---

    def test_evaluate_with_aislop_accepts_float_stdout(self) -> None:
        scores, task_passed, raw = evaluate_with_aislop(
            command=AISLOP_FLOAT_SCORE_COMMAND,
            evaluation_payload={"generated_output": "ignored", "case_id": "t"},
            timeout_seconds=10,
        )
        self.assertIsInstance(scores, dict)
        self.assertAlmostEqual(0.42, scores["total"])
        self.assertIsNone(task_passed)

    def test_evaluate_with_aislop_accepts_legacy_json_score(self) -> None:
        scores, task_passed, raw = evaluate_with_aislop(
            command=AISLOP_JSON_SCORE_COMMAND,
            evaluation_payload={"generated_output": "hello", "case_id": "t"},
            timeout_seconds=10,
        )
        self.assertIn("total", scores)
        self.assertIsInstance(scores["total"], float)

    def test_evaluate_with_aislop_accepts_multi_dimensional_scores(self) -> None:
        scores, task_passed, raw = evaluate_with_aislop(
            command=AISLOP_MULTI_SCORE_COMMAND,
            evaluation_payload={"generated_output": "hello", "case_id": "t"},
            timeout_seconds=10,
        )
        self.assertIn("correctness", scores)
        self.assertIn("style", scores)
        self.assertNotIn("total", scores)

    def test_evaluate_with_aislop_passes_structured_payload(self) -> None:
        """Evaluator receives a JSON payload with all task context fields."""
        # Command asserts required keys are present and returns 1.0 if so.
        verify_cmd = [
            sys.executable,
            "-c",
            (
                "import json,sys; p=json.loads(sys.stdin.read()); "
                "assert all(k in p for k in ('prompt','generated_output','expected_answer','case_id')); "
                "print('1.0')"
            ),
        ]
        scores, _, _ = evaluate_with_aislop(
            command=verify_cmd,
            evaluation_payload={
                "case_id": "c1",
                "task": "code_gen",
                "prompt": "hello",
                "system_prompt": None,
                "generated_output": "world",
                "expected_answer": "expected",
                "pass_condition": None,
                "pass_threshold": None,
            },
            timeout_seconds=10,
        )
        self.assertAlmostEqual(1.0, scores["total"])

    def test_evaluate_with_aislop_evaluator_can_return_task_passed(self) -> None:
        """Evaluator that returns task_passed: true is propagated to the trial row."""
        cmd = [
            sys.executable,
            "-c",
            "import json,sys; _ = sys.stdin.read(); print(json.dumps({'score': 0.9, 'task_passed': True}))",
        ]
        scores, task_passed, _ = evaluate_with_aislop(
            command=cmd,
            evaluation_payload={"generated_output": "x", "case_id": "t"},
            timeout_seconds=10,
        )
        self.assertAlmostEqual(0.9, scores["total"])
        self.assertTrue(task_passed)
        self.assertNotIn("task_passed", scores)

    # --- Task pass conditions ---

    def test_task_passed_exact_match(self) -> None:
        expected_output = "generated:Return hello world"  # what RUNNER_COMMAND produces
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "cases": [{
                        "id": "case-1",
                        "task": "code_gen",
                        "prompt": "Return hello world",
                        "expected_answer": expected_output,
                        "pass_condition": "exact_match",
                    }],
                }
            )
            results = run_benchmark(config)
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            self.assertTrue(trial_rows[0]["task_passed"])

    def test_task_passed_exact_match_fails_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "cases": [{
                        "id": "case-1",
                        "task": "code_gen",
                        "prompt": "Return hello world",
                        "expected_answer": "this will not match",
                        "pass_condition": "exact_match",
                    }],
                }
            )
            results = run_benchmark(config)
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            self.assertFalse(trial_rows[0]["task_passed"])

    def test_task_passed_score_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            # AISLOP_JSON_SCORE_COMMAND returns len(generated_output); output is long → high score
            config = BenchmarkConfig.from_dict(
                {
                    **_base_config(str(output_path)),
                    "cases": [{
                        "id": "case-1",
                        "task": "code_gen",
                        "prompt": "Return hello world",
                        "pass_condition": "score_threshold",
                        "pass_threshold": 0.0,  # always passes when score >= 0
                    }],
                }
            )
            results = run_benchmark(config)
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            self.assertTrue(trial_rows[0]["task_passed"])

    def test_task_pass_rate_in_aggregate(self) -> None:
        trials_with_task_passed = [
            {"row_type": "trial", "trial_index": i, "benchmark_name": "t",
             "case_id": "c1", "task": "code_gen", "run_id": None,
             "error": False,
             "ttft_seconds": 0.1, "total_seconds": 1.0,
             "tokens_per_second": 10.0, "decode_tokens_per_second": None,
             "quality_scores": {"total": 0.8},
             "task_passed": (i < 2),  # first 2 pass, third fails
             }
            for i in range(3)
        ]
        agg = aggregate_trials("c1", "code_gen", trials_with_task_passed)
        self.assertAlmostEqual(2 / 3, agg["task_pass_rate"])
        self.assertEqual(2, agg["task_passed_count"])

    # --- Run manifest ---

    def test_run_manifest_captured_in_results_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path)))
            results = run_benchmark(config)

            manifest = results[0]
            self.assertEqual("run_manifest", manifest["row_type"])
            self.assertIn("run_id", manifest)
            self.assertIn("config_hash", manifest)
            self.assertIn("started_at", manifest)
            self.assertIn("python_version", manifest)
            self.assertIn("platform_info", manifest)
            self.assertIn("cpu_count", manifest)
            self.assertIn("git_commit", manifest)  # may be None but field is present
            self.assertEqual("test-run", manifest["benchmark_name"])
            self.assertEqual(["case-1"], manifest["case_ids"])

            # First line of JSONL is the manifest
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            first = json.loads(lines[0])
            self.assertEqual("run_manifest", first["row_type"])

    def test_run_manifest_run_id_consistent_across_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path), trials=2))
            results = run_benchmark(config)
            run_id = results[0]["run_id"]
            for row in results:
                if row["row_type"] in ("trial", "aggregate"):
                    self.assertEqual(run_id, row.get("run_id"))

    # --- Warmup trials ---

    def test_warmup_trials_excluded_from_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(
                _base_config(str(output_path), trials=2, warmup_trials=1)
            )
            results = run_benchmark(config)

            warmup_rows = [r for r in results if r["row_type"] == "warmup"]
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            agg_rows = [r for r in results if r["row_type"] == "aggregate"]

            self.assertEqual(1, len(warmup_rows))
            self.assertEqual(2, len(trial_rows))
            self.assertEqual(1, len(agg_rows))
            # Aggregate sees only the 2 recorded trials
            self.assertEqual(2, agg_rows[0]["trial_count"])

    # --- Statistical controls in aggregates ---

    def test_aggregate_trials_statistics(self) -> None:
        fake_trials = [
            {
                "row_type": "trial",
                "trial_index": i,
                "benchmark_name": "test",
                "case_id": "c1",
                "task": "code_gen",
                "run_id": None,
                "error": False,
                "ttft_seconds": 0.1 * (i + 1),
                "total_seconds": 1.0 * (i + 1),
                "tokens_per_second": 100.0 / (i + 1),
                "decode_tokens_per_second": 110.0 / (i + 1),
                "quality_scores": {"total": float(i + 1)},
                "task_passed": None,
            }
            for i in range(3)
        ]
        agg = aggregate_trials("c1", "code_gen", fake_trials)
        self.assertEqual("aggregate", agg["row_type"])
        self.assertEqual(3, agg["trial_count"])
        self.assertEqual(3, agg["execution_success_count"])
        self.assertAlmostEqual(1.0, agg["execution_success_rate"])
        # Mean
        self.assertIsNotNone(agg["mean_ttft_seconds"])
        self.assertIsNotNone(agg["stddev_ttft_seconds"])
        # New: median, p95, min, max
        self.assertIsNotNone(agg["median_ttft_seconds"])
        self.assertIsNotNone(agg["p95_ttft_seconds"])
        self.assertIsNotNone(agg["min_ttft_seconds"])
        self.assertIsNotNone(agg["max_ttft_seconds"])
        # Quality aggregate
        self.assertIn("total", agg["quality_scores_aggregate"])
        self.assertAlmostEqual(2.0, agg["quality_scores_aggregate"]["total"]["mean"])
        self.assertIsNotNone(agg["quality_scores_aggregate"]["total"]["median"])

    # --- VRAM telemetry ---

    def test_vram_sampler_computes_peak_and_average_with_baseline(self) -> None:
        # Values: first call is the baseline, subsequent calls are samples.
        baseline_val = 100.0
        sample_vals = [200.0, 150.0, 150.0, 150.0]
        all_vals = iter([baseline_val] + sample_vals + [150.0] * 20)
        sampler = VramSampler(sample_fn=lambda: next(all_vals, 150.0), interval_seconds=0.01)
        self.assertTrue(sampler.available)
        sampler.start()
        time.sleep(0.06)
        telemetry = sampler.stop()

        self.assertIsNotNone(telemetry["vram_baseline_mb"])
        self.assertIsNotNone(telemetry["vram_peak_mb"])
        self.assertIsNotNone(telemetry["vram_peak_delta_mb"])
        self.assertIsNotNone(telemetry["vram_avg_delta_mb"])
        # Baseline was captured before polling started
        self.assertEqual(baseline_val, telemetry["vram_baseline_mb"])
        # Peak is max of polling samples (200.0)
        self.assertEqual(200.0, telemetry["vram_peak_mb"])
        # Delta = peak - baseline
        self.assertAlmostEqual(200.0 - baseline_val, telemetry["vram_peak_delta_mb"])
        # Peak >= avg
        self.assertGreaterEqual(telemetry["vram_peak_mb"], telemetry["vram_baseline_mb"])

    def test_vram_sampler_unavailable_reports_none(self) -> None:
        sampler = VramSampler(sample_fn=None, interval_seconds=0.01)
        sampler._sample_fn = None  # noqa: SLF001 (test-only override)
        self.assertFalse(sampler.available)
        sampler.start()
        telemetry = sampler.stop()
        self.assertIsNone(telemetry["vram_baseline_mb"])
        self.assertIsNone(telemetry["vram_peak_mb"])
        self.assertIsNone(telemetry["vram_peak_delta_mb"])
        self.assertIsNone(telemetry["vram_avg_delta_mb"])

    # --- Subprocess runner ---

    def test_benchmark_case_decode_tps_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            config = BenchmarkConfig.from_dict(_base_config(str(output_path)))
            results = run_benchmark(config)
            trial_rows = [r for r in results if r["row_type"] == "trial"]
            row = trial_rows[0]
            # decode_tokens_per_second may be None (when ttft≈total) or a positive float
            decode_tps = row["decode_tokens_per_second"]
            self.assertTrue(decode_tps is None or decode_tps > 0)

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

    # --- HTTP runner (non-streaming) ---

    def test_run_prompt_http_uses_api_reported_token_count(self) -> None:
        response = {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"completion_tokens": 7, "prompt_tokens": 3},
        }
        with _fake_openai_server(response) as url:
            output, ttft_seconds, total_seconds, comp_tokens, prmpt_tokens = run_prompt_http(
                url=url,
                model_id="test-model",
                prompt="hi",
                parameters={},
                seed=None,
                max_tokens=None,
                timeout_seconds=5,
            )
        self.assertEqual("hello world", output)
        self.assertEqual(7, comp_tokens)
        self.assertEqual(3, prmpt_tokens)
        self.assertGreaterEqual(total_seconds, 0)
        # Non-streaming: TTFT equals total
        self.assertEqual(ttft_seconds, total_seconds)

    def test_run_prompt_http_falls_back_when_usage_missing(self) -> None:
        response = {"choices": [{"message": {"content": "no usage field here"}}]}
        with _fake_openai_server(response) as url:
            _, _, _, comp_tokens, prmpt_tokens = run_prompt_http(
                url=url,
                model_id="test-model",
                prompt="hi",
                parameters={},
                seed=None,
                max_tokens=None,
                timeout_seconds=5,
            )
        self.assertIsNone(comp_tokens)
        self.assertIsNone(prmpt_tokens)


if __name__ == "__main__":
    unittest.main()
