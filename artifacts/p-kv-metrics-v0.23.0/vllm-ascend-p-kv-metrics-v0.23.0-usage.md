# vLLM Ascend v0.23.0：P 实例 KV 指标使用说明

## 功能和边界

新补丁默认关闭。开启后只在 `kv_role=kv_producer` 的 P 实例收集：

- `p_kv_hit`：请求 prompt token 数和 P 侧去重 KV 前缀命中数；
- `p_kv_read`：计划从外部 KV Pool 读取的 token 范围，以及真实 backend
  `get`/layerwise G2L copy 调用的墙钟耗时；
- `p_kv_config`：block size、传输粒度和 worker rank 等解释信息。

`hit_tokens=min(request_tokens,max(local_hit_tokens,pool_hit_tokens))`，本地
HBM 和 Pool 的重叠前缀不会相加。

`scheduled_read_tokens` 是计划读取范围，不是确认成功读取的 token 数。
`read_time_ns` 不包含 lookup、排队、调度和 prefill，也不是纯 HBM 访问时间；
如果 backend 在异步工作完成前返回，计时在返回点结束。

## 开启与关闭

在启动 P 实例前设置：

```bash
export VLLM_ASCEND_KV_METRICS=1
export VLLM_ASCEND_KV_METRICS_DIR=/data/logs/vllm-kv-metrics/run-001
```

然后重启 P 实例。建议每次测试使用一个全新的目录，避免混入历史数据。
目录必须允许 scheduler 和所有 worker 进程写入。容器/Kubernetes 环境应挂载
持久卷；跨节点 TP 要在测试后把所有节点文件收集到同一目录。

关闭时：

```bash
export VLLM_ASCEND_KV_METRICS=0
```

或者删除这两个环境变量并重启。默认关闭时不会创建指标文件，也不会执行读取计时。

## 输出文件

默认目录是 `/tmp/vllm_ascend_kv_metrics`。指定目录后会产生：

```text
p_kv_metrics.scheduler.<hostname>.<pid>.jsonl
p_kv_metrics.worker.<hostname>.<pid>.jsonl
```

文件为 append-only JSONL，vLLM 不负责轮转。普通 vLLM 日志中只会显示一次
“metrics enabled”和实际输出路径，不再混入逐请求指标行。

检查是否生效：

```bash
find /data/logs/vllm-kv-metrics/run-001 \
  -name 'p_kv_metrics.*.jsonl' -type f -print
```

查看命中记录：

```bash
grep '"event":"p_kv_hit"' \
  /data/logs/vllm-kv-metrics/run-001/p_kv_metrics.scheduler.*.jsonl
```

查看读取记录：

```bash
grep '"event":"p_kv_read"' \
  /data/logs/vllm-kv-metrics/run-001/p_kv_metrics.worker.*.jsonl
```

## 最小验证

通过现有网关发送两次 prompt 完全相同且足够长的请求：

1. 第一次建立 KV cache，通常为冷请求；
2. 等第一次完成并确认 KV 已保存；
3. 再次发送相同 prompt；
4. 用 `request_id` 对齐 scheduler 的 hit 事件和 worker 的 read 事件。

受 block/transfer granularity 影响，Pool 命中通常按完整传输粒度计数；
`discard_partial_chunks=True` 时，尾部不足一个粒度的 token 不计 Pool 命中。

## 汇总脚本

测试结束后，在打过补丁的 vLLM Ascend 源码目录执行：

```bash
python tools/extract_p_kv_metrics.py \
  --input-dir /data/logs/vllm-kv-metrics/run-001 \
  --output-dir /data/logs/vllm-kv-metrics/run-001/report
```

生成：

```text
report/p_kv_request_metrics.csv
report/p_kv_summary.json
```

CSV 中的重要字段：

- `request_tokens`、`hit_tokens`、`hit_ratio`；
- `scheduled_read_tokens`；
- `read_time_ms_max_rank`：该请求各 rank 观测读取调用时间的最大值，作为请求侧
  读取延迟口径；
- `read_time_ms_sum_ranks`：rank 时间之和，仅供工作量诊断，不能当请求延迟；
- `read_outcome`：`ok`、`failed`、`empty` 或 `missing`；
- `shared_batch_timing=True`：此次计时来自多请求共享的 batch，不能完全归因给
  某一个请求。

JSON 汇总包含：

- 请求数、输入 token 总数、命中 token 总数；
- 加权命中率 `sum(hit_tokens)/sum(request_tokens)`；
- 成功/失败/空读/缺失读取记录数量；
- 成功读取调用的 total、mean、p50、p95、p99 和 max。

若 `hit_request_count` 与压测请求数不一致，优先检查是否收齐所有 P scheduler
文件，以及请求是否在进入 allocation 前终止/抢占。若 `read_request_count` 小于
hit 请求数，可能是该请求只有本地 HBM 命中、不需要外部读取，也可能是 worker
文件没有收齐。
