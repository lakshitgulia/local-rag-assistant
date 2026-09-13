"""
Top-3 citation accuracy eval. Takes a list of {question, expected_source_file}
pairs, runs each through the running app's /api/query, and checks whether
expected_source_file appears anywhere in the top-3 returned citations.

Usage:
    python eval.py questions.json
    python eval.py questions.json --url http://localhost:8000 --role all

questions.json format:
    [
      {"question": "...", "expected_source_file": "some_file.pdf"},
      ...
    ]
"""
import argparse
import json
import sys

import requests


def run_eval(questions: list, url: str, role: str) -> list:
    results = []
    for item in questions:
        question = item["question"]
        expected = item["expected_source_file"]
        resp = requests.post(f"{url}/api/query", json={"question": question, "role": role}, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        cited_files = [s["file"] for s in data["sources"]]
        passed = expected in cited_files
        results.append({
            "question": question,
            "expected_source_file": expected,
            "cited_files": cited_files,
            "grounded": data["grounded"],
            "response_time_seconds": data["response_time_seconds"],
            "passed": passed,
        })
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {question}")
        print(f"       expected: {expected}")
        print(f"       cited:    {cited_files or '(none)'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("questions_file", help="Path to a JSON file of {question, expected_source_file} pairs")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--role", default="all")
    args = parser.parse_args()

    with open(args.questions_file) as f:
        questions = json.load(f)

    results = run_eval(questions, args.url, args.role)

    correct = sum(1 for r in results if r["passed"])
    total = len(results)
    pct = (correct / total * 100) if total else 0.0

    print()
    print(f"{correct}/{total} correct ({pct:.0f}%)")

    if correct < total:
        print("\nFailed questions:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['question']!r} -> expected {r['expected_source_file']!r}, got {r['cited_files']}")

    sys.exit(0 if correct == total else 1)
