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

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs


class KVMetricWriter:
    """Write one P-side KV metric event per JSONL line."""

    def __init__(self, component: str) -> None:
        output_dir = Path(envs.VLLM_ASCEND_KV_METRICS_DIR).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.path = output_dir / f"p_kv_metrics.{component}.{self.hostname}.{self.pid}.jsonl"
        self._lock = threading.Lock()
        self._write_failed = False

    def write(self, event: str, **fields: Any) -> None:
        if self._write_failed:
            return
        record = {
            "timestamp_ns": time.time_ns(),
            "event": event,
            "hostname": self.hostname,
            "pid": self.pid,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as output:
                output.write(line)
        except OSError:
            self._write_failed = True
            logger.exception("Disabling P-side KV metrics after failing to write %s", self.path)


def create_p_kv_metric_writer(component: str, kv_role: str) -> KVMetricWriter | None:
    if kv_role != "kv_producer" or not envs.VLLM_ASCEND_KV_METRICS:
        return None
    return KVMetricWriter(component)


def get_scheduled_read_tokens(request: Any) -> int:
    load_spec = request.load_spec
    if load_spec is None:
        return 0
    read_end = int(
        load_spec.token_len
        or load_spec.kvpool_store_skip_tokens
        or load_spec.kvpool_cached_tokens
    )
    return max(read_end - int(load_spec.vllm_cached_tokens), 0)
