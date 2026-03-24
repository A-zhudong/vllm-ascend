import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_client import Counter, Histogram

from vllm_ascend import envs

KVPOOL_LOAD_SECONDS = Histogram(
    "vllm_ascend_kvpool_load_seconds",
    "KV Pool load latency in seconds.",
)
KVPOOL_LOAD_TPOT_SECONDS = Histogram(
    "vllm_ascend_kvpool_load_tpot_seconds",
    "KV Pool load TPOT in seconds (load_seconds / hit_tokens).",
)
KVPOOL_HIT_TOKENS_TOTAL = Counter(
    "vllm_ascend_kvpool_hit_tokens_total",
    "Total KV Pool hit tokens.",
)
KVPOOL_RECOMPUTE_TOKENS_TOTAL = Counter(
    "vllm_ascend_kvpool_recompute_tokens_total",
    "Total KV Pool recompute tokens.",
)
KVPOOL_PROFILED_REQUESTS_TOTAL = Counter(
    "vllm_ascend_kvpool_profiled_requests_total",
    "Total profiled KV Pool requests emitted.",
)


@dataclass
class _RequestStats:
    request_id: str
    prompt_tokens_total: int
    hit_tokens: int
    recompute_tokens: int
    load_time_ns: int = 0
    emitted: bool = False


class KVPoolRequestProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, _RequestStats] = {}

    def record_lookup(self, request_id: str | None, prompt_tokens_total: int | None, hit_tokens: int | None) -> None:
        req_id = _normalize_request_id(request_id)
        if req_id is None:
            return
        prompt = _safe_non_negative_int(prompt_tokens_total)
        hit = _safe_non_negative_int(hit_tokens)
        if prompt is None or hit is None:
            return
        hit = min(hit, prompt)
        with self._lock:
            self._stats[req_id] = _RequestStats(
                request_id=req_id,
                prompt_tokens_total=prompt,
                hit_tokens=hit,
                recompute_tokens=max(prompt - hit, 0),
            )

    def add_load_time_ns(self, request_id: str | None, load_time_ns: int | None) -> None:
        req_id = _normalize_request_id(request_id)
        load_ns = _safe_non_negative_int(load_time_ns)
        if req_id is None or load_ns is None:
            return
        with self._lock:
            stats = self._stats.get(req_id)
            if stats is None:
                stats = _RequestStats(
                    request_id=req_id,
                    prompt_tokens_total=0,
                    hit_tokens=0,
                    recompute_tokens=0,
                )
                self._stats[req_id] = stats
            stats.load_time_ns += load_ns

    def emit_if_ready(self, request_id: str | None) -> None:
        req_id = _normalize_request_id(request_id)
        if req_id is None:
            return
        with self._lock:
            stats = self._stats.get(req_id)
            if stats is None or stats.emitted:
                return
            if stats.hit_tokens > 0 and stats.load_time_ns <= 0:
                return
            payload = {
                "request_id": stats.request_id,
                "prompt_tokens_total": stats.prompt_tokens_total,
                "hit_tokens": stats.hit_tokens,
                "recompute_tokens": stats.recompute_tokens,
                "load_time_ms": stats.load_time_ns / 1_000_000,
            }
            self._emit_locked(payload, stats)
            stats.emitted = True

    def pop(self, request_id: str | None) -> None:
        req_id = _normalize_request_id(request_id)
        if req_id is None:
            return
        with self._lock:
            self._stats.pop(req_id, None)

    def _emit_locked(self, payload: dict[str, Any], stats: _RequestStats) -> None:
        log_path = Path(envs.KVPOOL_VLLM_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        load_seconds = stats.load_time_ns / 1_000_000_000
        KVPOOL_LOAD_SECONDS.observe(load_seconds)
        KVPOOL_HIT_TOKENS_TOTAL.inc(stats.hit_tokens)
        KVPOOL_RECOMPUTE_TOKENS_TOTAL.inc(stats.recompute_tokens)
        KVPOOL_PROFILED_REQUESTS_TOTAL.inc()
        if stats.hit_tokens > 0:
            KVPOOL_LOAD_TPOT_SECONDS.observe(load_seconds / stats.hit_tokens)


def _normalize_request_id(request_id: Any) -> str | None:
    if request_id is None:
        return None
    request_id_str = str(request_id)
    if request_id_str == "":
        return None
    return request_id_str


def _safe_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if result < 0:
        return None
    return result


def extract_request_id(obj: Any = None, kwargs: dict[str, Any] | None = None) -> str | None:
    if kwargs is not None and "request_id" in kwargs:
        req_id = _normalize_request_id(kwargs.get("request_id"))
        if req_id is not None:
            return req_id
    if obj is None:
        return None

    for attr in ("request_id", "_request_id", "req_id"):
        req_id = _normalize_request_id(getattr(obj, attr, None))
        if req_id is not None:
            return req_id

    connector_meta = getattr(obj, "connector_meta", None)
    if connector_meta is not None:
        req_id = _normalize_request_id(getattr(connector_meta, "request_id", None))
        if req_id is not None:
            return req_id
    return None


kvpool_request_profiler = KVPoolRequestProfiler()
