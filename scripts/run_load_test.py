"""
Load test runner — executes Locust headlessly and parses results.

Starts the FastAPI server if not already running, runs Locust with
configurable parameters, parses the output JSON, checks SLA thresholds,
and saves a human-readable report to docs/.

Usage:
    python scripts/run_load_test.py                    # full test (u=10, t=60s)
    python scripts/run_load_test.py --users 20 --time 120
    python scripts/run_load_test.py --smoke            # 1 user, 15s
    python scripts/run_load_test.py --parse-only       # parse saved locust_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.config import settings

LOCUST_RESULTS_FILE = Path("docs/locust_results.json")
LOCUST_REPORT_FILE  = Path("docs/load_test_report.txt")
SLA_P95_MS          = 3000.0   # 3 seconds
SLA_ERROR_RATE      = 0.01     # 1% max error rate


def check_api_live(host: str = "http://localhost:8000", timeout: int = 30) -> bool:
    """Poll /health until the API responds or timeout."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{host}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_locust_headless(
    host: str = "http://localhost:8000",
    users: int = 10,
    spawn_rate: int = 2,
    run_time: str = "60s",
    results_file: Path = LOCUST_RESULTS_FILE,
) -> bool:
    """
    Run Locust headlessly. Returns True if the process exits cleanly.
    Results are written to results_file as JSON (via --json flag).
    """
    results_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "locust",
        "-f", "locustfile.py",
        "--headless",
        "--host", host,
        "-u", str(users),
        "-r", str(spawn_rate),
        "-t", run_time,
        "--json",
        "--logfile", str(results_file.with_suffix(".log")),
        "--html", str(results_file.with_suffix(".html")),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Locust writes JSON stats to stdout with --json
    if result.stdout.strip():
        try:
            data = json.loads(result.stdout.strip())
            results_file.write_text(json.dumps(data, indent=2))
            logger.success(f"Locust results saved → {results_file}")
        except json.JSONDecodeError:
            # Locust may write partial JSON; save raw output
            results_file.write_text(result.stdout)
    if result.returncode != 0:
        logger.warning(f"Locust exited with code {result.returncode}")
        logger.warning(result.stderr[-2000:])
    return result.returncode == 0


def parse_locust_results(results_file: Path = LOCUST_RESULTS_FILE) -> dict:
    """
    Parse Locust JSON results and extract key metrics.
    Returns dict with p50, p95, p99, error_rate, num_requests per endpoint.
    """
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return {}

    raw = json.loads(results_file.read_text())

    # Locust JSON format varies by version; handle both list and dict forms
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = raw.get("stats", [])
    else:
        logger.error(f"Unexpected JSON format in {results_file}")
        return {}

    parsed = {}
    for entry in entries:
        name = entry.get("name") or entry.get("Name", "")
        if name == "Aggregated":
            name = "__total__"
        parsed[name] = {
            "num_requests": entry.get("num_requests") or entry.get("Requests") or 0,
            "num_failures": entry.get("num_failures") or entry.get("Failures") or 0,
            "p50_ms": entry.get("50%") or entry.get("response_times", {}).get("50", 0),
            "p75_ms": entry.get("75%") or entry.get("response_times", {}).get("75", 0),
            "p95_ms": entry.get("95%") or entry.get("response_times", {}).get("95", 0),
            "p99_ms": entry.get("99%") or entry.get("response_times", {}).get("99", 0),
            "avg_ms": entry.get("avg_response_time") or entry.get("Average Response Time") or 0,
            "rps": entry.get("requests_per_s") or entry.get("Requests/s") or 0,
        }
        reqs = parsed[name]["num_requests"]
        fails = parsed[name]["num_failures"]
        parsed[name]["error_rate"] = fails / reqs if reqs > 0 else 0.0

    return parsed


def check_sla(parsed: dict) -> tuple[bool, list[str]]:
    """
    Check SLA thresholds. Returns (passed, violations).
    SLA: p95 < 3000ms AND error_rate < 1% for /query endpoint.
    """
    violations = []
    query_stats = parsed.get("/query") or parsed.get("__total__", {})

    p95 = query_stats.get("p95_ms", 0)
    err = query_stats.get("error_rate", 0)

    if p95 > SLA_P95_MS:
        violations.append(f"P95 {p95:.0f}ms > {SLA_P95_MS:.0f}ms SLA")
    if err > SLA_ERROR_RATE:
        violations.append(f"Error rate {err:.2%} > {SLA_ERROR_RATE:.0%} SLA")

    return len(violations) == 0, violations


def generate_text_report(
    parsed: dict,
    sla_passed: bool,
    violations: list[str],
    test_config: dict,
    output_path: Path = LOCUST_REPORT_FILE,
) -> None:
    """Write a human-readable load test report."""
    lines = [
        "═" * 68,
        "  Clinical QA System — Load Test Report",
        "  LJMU MSc ML&AI | Amit Choudhary (1187247)",
        "═" * 68,
        "",
        f"  Test configuration:",
        f"    Host:        {test_config.get('host', 'http://localhost:8000')}",
        f"    Concurrency: {test_config.get('users', 10)} users",
        f"    Spawn rate:  {test_config.get('spawn_rate', 2)} users/s",
        f"    Duration:    {test_config.get('run_time', '60s')}",
        "",
        "  Results by endpoint:",
        f"  {'Endpoint':<20} {'Req':>6} {'Fail':>6} {'p50':>7} {'p95':>7} {'p99':>7} {'RPS':>7} {'Err%':>6}",
        "  " + "─" * 65,
    ]
    for name, stats in sorted(parsed.items()):
        if name == "__total__":
            continue
        lines.append(
            f"  {name:<20} "
            f"{stats['num_requests']:>6} "
            f"{stats['num_failures']:>6} "
            f"{stats['p50_ms']:>6.0f}ms "
            f"{stats['p95_ms']:>6.0f}ms "
            f"{stats['p99_ms']:>6.0f}ms "
            f"{stats['rps']:>6.1f} "
            f"{stats['error_rate']:>5.1%}"
        )
    if "__total__" in parsed:
        t = parsed["__total__"]
        lines += [
            "  " + "─" * 65,
            f"  {'TOTAL':<20} "
            f"{t['num_requests']:>6} "
            f"{t['num_failures']:>6} "
            f"{t['p50_ms']:>6.0f}ms "
            f"{t['p95_ms']:>6.0f}ms "
            f"{t['p99_ms']:>6.0f}ms "
            f"{t['rps']:>6.1f} "
            f"{t['error_rate']:>5.1%}",
        ]
    lines += [
        "",
        f"  SLA: P95 < {SLA_P95_MS:.0f}ms AND error rate < {SLA_ERROR_RATE:.0%}",
        f"  Verdict: {'✅ PASS' if sla_passed else '❌ FAIL'}",
    ]
    if violations:
        lines.append("  Violations:")
        for v in violations:
            lines.append(f"    - {v}")
    lines += ["", "═" * 68]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print("\n".join(lines))
    logger.info(f"Load test report saved → {output_path}")


def smoke_test(host: str = "http://localhost:8000") -> bool:
    """Single request smoke test — validates API is responding."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "question": "What is metformin used for?",
        "use_rag": False,
        "use_hallucination_filter": False,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{host}/query",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as r:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            body = json.loads(r.read())
            ok = "answer" in body and elapsed_ms < SLA_P95_MS
            print(f"\n  Smoke test: {'✅ PASS' if ok else '❌ FAIL'}")
            print(f"    Response time: {elapsed_ms:.0f}ms (SLA: {SLA_P95_MS:.0f}ms)")
            print(f"    Answer preview: {body.get('answer', '')[:80]}…")
            return ok
    except Exception as e:
        print(f"\n  Smoke test: ❌ FAIL — {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test runner for Clinical QA API")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--spawn-rate", type=int, default=2)
    parser.add_argument("--time", dest="run_time", default="60s")
    parser.add_argument("--smoke", action="store_true", help="Single request smoke test")
    parser.add_argument("--parse-only", action="store_true",
                        help="Parse existing results file, no new test")
    args = parser.parse_args()

    settings.ensure_dirs()

    if args.smoke:
        ok = smoke_test(args.host)
        sys.exit(0 if ok else 1)

    if args.parse_only:
        parsed = parse_locust_results()
        if parsed:
            sla_ok, violations = check_sla(parsed)
            generate_text_report(parsed, sla_ok, violations, vars(args))
        sys.exit(0)

    # Full load test
    logger.info(f"Checking API is live at {args.host}…")
    if not check_api_live(args.host):
        logger.error(f"API not reachable at {args.host}. Start it with: make run-api")
        sys.exit(1)

    logger.info(f"Starting load test: {args.users} users × {args.run_time}")
    run_locust_headless(
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        run_time=args.run_time,
    )

    parsed = parse_locust_results()
    if parsed:
        sla_ok, violations = check_sla(parsed)
        generate_text_report(parsed, sla_ok, violations, vars(args))
        sys.exit(0 if sla_ok else 1)


if __name__ == "__main__":
    main()
