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

import json
import threading
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from tests.ut.distributed.ascend_store.test_kv_transfer import (
    FakeStore,
    FakeTokenDatabase,
)
from tools.extract_p_kv_metrics import summarize_records
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
    kv_metrics as kv_metrics_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
    kv_transfer as kv_transfer_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
    pool_worker as pool_worker_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_metrics import (
    create_p_kv_metric_writer,
    get_scheduled_read_tokens,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
    KVTransferThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)
# isort: on


def _request(
    vllm_cached_tokens: int = 32,
    kvpool_cached_tokens: int = 96,
    token_len: int = 96,
) -> ReqMeta:
    return ReqMeta(
        req_id="request-1",
        token_len_chunk=token_len,
        block_ids=[0, 1],
        block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
        load_spec=LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=kvpool_cached_tokens,
            can_load=True,
            token_len=token_len,
        ),
    )


def test_metrics_default_to_disabled():
    with patch.object(kv_metrics_module.envs, "VLLM_ASCEND_KV_METRICS", False):
        assert create_p_kv_metric_writer("scheduler", "kv_producer") is None


def test_metric_writer_uses_dedicated_directory(tmp_path):
    with (
        patch.object(kv_metrics_module.envs, "VLLM_ASCEND_KV_METRICS", True),
        patch.object(kv_metrics_module.envs, "VLLM_ASCEND_KV_METRICS_DIR", str(tmp_path)),
    ):
        writer = create_p_kv_metric_writer("scheduler", "kv_producer")
        assert writer is not None
        writer.write("p_kv_hit", request_id="r1", request_tokens=64, hit_tokens=32)

    files = list(tmp_path.glob("p_kv_metrics.scheduler.*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["event"] == "p_kv_hit"
    assert record["request_id"] == "r1"
    assert record["hit_tokens"] == 32


def test_scheduled_read_tokens_use_raw_full_hit_before_scheduler_decrement():
    request = _request(vllm_cached_tokens=0, kvpool_cached_tokens=63, token_len=64)
    request.load_spec.token_len = 0  # type: ignore[union-attr]
    request.load_spec.kvpool_store_skip_tokens = 64  # type: ignore[union-attr]

    assert get_scheduled_read_tokens(request) == 64


def test_layerwise_read_time_accumulates_and_writes_once():
    writer = MagicMock()
    thread = KVTransferThread(
        m_store=MagicMock(),
        token_database=MagicMock(group_block_len={0: [16]}),
        block_size=16,
        tp_rank=2,
        ready_event=threading.Event(),
        kv_metric_writer=writer,
        kv_metric_dp_rank=3,
        kv_metric_pp_rank=1,
    )
    request = _request()

    thread.record_kv_read_metrics([request], 1_000_000, "layerwise_gva", "ok", final=False)
    thread.record_kv_read_metrics([request], 2_000_000, "layerwise_gva", "ok", final=True)

    writer.write.assert_called_once_with(
        "p_kv_read",
        request_id="request-1",
        scheduled_read_tokens=64,
        read_time_ns=3_000_000,
        mode="layerwise_gva",
        outcome="ok",
        dp_rank=3,
        pp_rank=1,
        tp_rank=2,
        read_call_count=2,
        max_batch_size=1,
        timing_scope="request",
    )
    assert thread._kv_read_elapsed_ns == {}


def test_layerwise_failure_and_shared_batch_scope_are_preserved():
    writer = MagicMock()
    thread = KVTransferThread(
        m_store=MagicMock(),
        token_database=MagicMock(group_block_len={0: [16]}),
        block_size=16,
        tp_rank=0,
        kv_metric_writer=writer,
    )
    request_1 = _request()
    request_2 = _request()
    request_2.req_id = "request-2"

    thread.record_kv_read_metrics([request_1, request_2], 1_000_000, "layerwise_gva", "failed", final=False)
    thread.record_kv_read_metrics([request_1, request_2], 2_000_000, "layerwise_gva", "ok", final=True)

    assert writer.write.call_count == 2
    for call in writer.write.call_args_list:
        assert call.kwargs["outcome"] == "failed"
        assert call.kwargs["read_time_ns"] == 3_000_000
        assert call.kwargs["max_batch_size"] == 2
        assert call.kwargs["timing_scope"] == "shared_batch"


def test_async_read_records_backend_time_for_request():
    writer = MagicMock()
    store = FakeStore()
    store.get = MagicMock(return_value=[0, 0])
    thread = KVCacheStoreRecvingThread(
        m_store=store,
        token_database=FakeTokenDatabase(),
        block_size=16,
        tp_rank=0,
        dcp_size=1,
        ready_event=threading.Event(),
        invalid_block_ids=set(),
        invalid_block_ids_lock=threading.Lock(),
        kv_metric_writer=writer,
    )
    request = _request(vllm_cached_tokens=0, kvpool_cached_tokens=32, token_len=32)
    thread.request_queue.put(request)

    with patch.object(kv_transfer_module.time, "perf_counter_ns", side_effect=[100, 5_000_100]):
        thread._handle_request(request)

    assert writer.write.call_args.kwargs["read_time_ns"] == 5_000_000
    assert writer.write.call_args.kwargs["mode"] == "async"
    assert writer.write.call_args.kwargs["outcome"] == "ok"


def test_sync_read_records_backend_time_for_request():
    writer = MagicMock()
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.current_layer = 0
    worker.use_layerwise = False
    worker.group_uses_align_state = [False]
    worker.cache_transfer_granularity = 16
    worker.load_async = False
    worker.token_database = FakeTokenDatabase()
    worker.grouped_block_size = [16]
    worker.m_store = MagicMock()
    worker.m_store.get.return_value = [0, 0]
    worker._invalid_block_ids = set()
    worker.tp_rank = 0
    worker.dp_rank = 2
    worker.pp_rank = 1
    worker._kv_metric_writer = writer
    request = _request(vllm_cached_tokens=0, kvpool_cached_tokens=32, token_len=32)
    metadata = AscendConnectorMetadata(set(), set())
    metadata.add_request(request)

    with patch.object(pool_worker_module.time, "perf_counter_ns", side_effect=[100, 7_000_100]):
        worker.start_load_kv(metadata)

    assert writer.write.call_args.kwargs["read_time_ns"] == 7_000_000
    assert writer.write.call_args.kwargs["mode"] == "sync"
    assert writer.write.call_args.kwargs["outcome"] == "ok"


def test_disabled_read_metrics_do_not_call_timer():
    store = FakeStore()
    store.get = MagicMock(return_value=[0, 0])
    thread = KVCacheStoreRecvingThread(
        m_store=store,
        token_database=FakeTokenDatabase(),
        block_size=16,
        tp_rank=0,
        dcp_size=1,
        ready_event=threading.Event(),
        invalid_block_ids=set(),
        invalid_block_ids_lock=threading.Lock(),
    )
    request = _request(vllm_cached_tokens=0, kvpool_cached_tokens=32, token_len=32)
    thread.request_queue.put(request)

    with patch.object(kv_transfer_module.time, "perf_counter_ns") as timer:
        thread._handle_request(request)

    timer.assert_not_called()


def test_extractor_uses_max_rank_time_and_weighted_hit_ratio():
    rows, summary = summarize_records(
        [
            {
                "timestamp_ns": 1,
                "event": "p_kv_hit",
                "request_id": "r1",
                "request_tokens": 100,
                "hit_tokens": 80,
            },
            {
                "timestamp_ns": 2,
                "event": "p_kv_read",
                "request_id": "r1",
                "scheduled_read_tokens": 80,
                "read_time_ns": 2_000_000,
                "outcome": "ok",
                "mode": "sync",
                "max_batch_size": 1,
            },
            {
                "timestamp_ns": 3,
                "event": "p_kv_read",
                "request_id": "r1",
                "scheduled_read_tokens": 80,
                "read_time_ns": 3_000_000,
                "outcome": "ok",
                "mode": "sync",
                "max_batch_size": 1,
            },
        ]
    )

    assert rows[0]["read_time_ms_max_rank"] == 3.0
    assert rows[0]["read_time_ms_sum_ranks"] == 5.0
    assert summary["weighted_hit_ratio"] == 0.8
