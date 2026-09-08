#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hit_events: dict[str, dict[str, Any]] = {}
    read_events: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_hit_events = 0

    for record in records:
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        if record.get("event") == "p_kv_hit":
            if request_id in hit_events:
                duplicate_hit_events += 1
            if request_id not in hit_events or int(record.get("timestamp_ns", 0)) >= int(
                hit_events[request_id].get("timestamp_ns", 0)
            ):
                hit_events[request_id] = record
        elif record.get("event") == "p_kv_read":
            read_events[request_id].append(record)

    rows: list[dict[str, Any]] = []
    for request_id in sorted(set(hit_events) | set(read_events)):
        hit = hit_events.get(request_id, {})
        reads = read_events.get(request_id, [])
        read_times_ms = [max(int(item.get("read_time_ns", 0)), 0) / 1_000_000 for item in reads]
        outcomes = {str(item.get("outcome", "unknown")) for item in reads}
        if "failed" in outcomes:
            read_outcome = "failed"
        elif "ok" in outcomes:
            read_outcome = "ok"
        elif outcomes == {"empty"}:
            read_outcome = "empty"
        else:
            read_outcome = "missing"

        request_tokens = hit.get("request_tokens")
        hit_tokens = hit.get("hit_tokens")
        hit_ratio = (
            int(hit_tokens) / int(request_tokens)
            if request_tokens is not None and hit_tokens is not None and int(request_tokens) > 0
            else None
        )
        rows.append(
            {
                "request_id": request_id,
                "request_tokens": request_tokens,
                "hit_tokens": hit_tokens,
                "hit_ratio": hit_ratio,
                "scheduled_read_tokens": max(
                    (int(item.get("scheduled_read_tokens", 0)) for item in reads),
                    default=None,
                ),
                "read_time_ms_max_rank": max(read_times_ms, default=None),
                "read_time_ms_sum_ranks": sum(read_times_ms) if reads else None,
                "read_rank_events": len(reads),
                "read_outcome": read_outcome,
                "read_modes": "|".join(sorted({str(item.get("mode", "unknown")) for item in reads})),
                "shared_batch_timing": any(item.get("timing_scope") == "shared_batch" for item in reads),
                "max_batch_size": max((int(item.get("max_batch_size", 0)) for item in reads), default=None),
            }
        )

    total_request_tokens = sum(int(row["request_tokens"]) for row in rows if row["request_tokens"] is not None)
    total_hit_tokens = sum(int(row["hit_tokens"]) for row in rows if row["hit_tokens"] is not None)
    ok_read_times = [
        float(row["read_time_ms_max_rank"])
        for row in rows
        if row["read_outcome"] == "ok" and row["read_time_ms_max_rank"] is not None
    ]
    summary = {
        "hit_request_count": len(hit_events),
        "read_request_count": len(read_events),
        "joined_request_count": len(set(hit_events) & set(read_events)),
        "duplicate_hit_event_count": duplicate_hit_events,
        "total_request_tokens": total_request_tokens,
        "total_hit_tokens": total_hit_tokens,
        "weighted_hit_ratio": total_hit_tokens / total_request_tokens if total_request_tokens else None,
        "read_outcome_counts": {
            outcome: sum(row["read_outcome"] == outcome for row in rows)
            for outcome in ("ok", "failed", "empty", "missing")
        },
        "ok_read_time_ms_max_rank": {
            "count": len(ok_read_times),
            "total": sum(ok_read_times),
            "mean": sum(ok_read_times) / len(ok_read_times) if ok_read_times else None,
            "p50": _percentile(ok_read_times, 0.50),
            "p95": _percentile(ok_read_times, 0.95),
            "p99": _percentile(ok_read_times, 0.99),
            "max": max(ok_read_times, default=None),
        },
        "interpretation": {
            "hit_tokens": "Deduplicated P-side prefix frontier: max(local HBM hit, KV Pool hit).",
            "scheduled_read_tokens": "Planned external transfer range, not confirmed successful tokens.",
            "read_time_ms_max_rank": "Maximum observed external read/copy call time across ranks for the request.",
            "shared_batch_timing": (
                "True means the measured call served multiple requests and is not exclusively attributable."
            ),
        },
    }
    return rows, summary


def load_records(input_dir: Path) -> tuple[list[dict[str, Any]], list[str], int]:
    paths = sorted(input_dir.rglob("p_kv_metrics.*.jsonl"))
    records: list[dict[str, Any]] = []
    invalid_lines = 0
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                if isinstance(record, dict):
                    records.append(record)
                else:
                    invalid_lines += 1
    return records, [str(path) for path in paths], invalid_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate P-side KV hit and external-read JSONL metrics.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing p_kv_metrics.*.jsonl")
    parser.add_argument("--output-dir", type=Path, help="Report directory; defaults to <input-dir>/report")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    records, input_files, invalid_lines = load_records(args.input_dir)
    rows, summary = summarize_records(records)
    summary["input_files"] = input_files
    summary["invalid_jsonl_line_count"] = invalid_lines

    csv_path = output_dir / "p_kv_request_metrics.csv"
    fields = list(rows[0]) if rows else [
        "request_id",
        "request_tokens",
        "hit_tokens",
        "hit_ratio",
        "scheduled_read_tokens",
        "read_time_ms_max_rank",
        "read_time_ms_sum_ranks",
        "read_rank_events",
        "read_outcome",
        "read_modes",
        "shared_batch_timing",
        "max_batch_size",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "p_kv_summary.json"
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
