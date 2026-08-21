# fused_li_manage 算子

本仓库只保留 Ascend `fused_li_manage` 自定义算子的实现、构建代码，以及以下两类测试：

- `test_fused_li_manage.py`：正确性、缓存管理语义及 18-bit/21-bit 边界测试，同时提供简单延迟数据。
- `test_fused_li_manage_perf.py`：面向多 batch、序列长度和 miss 范围的性能测试。

两项测试均只调用 `fused_li_manage` 和 LightningIndexer 基线，不调用 Scatter 算子。

## 编译

在仓库根目录执行：

```bash
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
export NANOVLLM_CANN_BUILD_JOBS=64
export NANOVLLM_EXT_BUILD_JOBS=1
export SOC_VERSION=ascend910_9391
bash scripts/build_nanovllm_ops.sh
```

构建脚本只编译 `fused_li_manage`。主要产物为：

- `nanovllm/_C*.so`：Torch 算子注册扩展。
- `nanovllm/_cann_ops_custom/`：本地 CANN 自定义算子包。

## 测试环境

测试必须在 Ascend NPU 上运行。每次测试前可设置：

```bash
unset NANOVLLM_OFFLOAD_MODE
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_NUM_SPECULATIVE_TOKENS
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=11
```

`ASCEND_RT_VISIBLE_DEVICES=11` 会把物理设备 11 映射为测试命令中的 `npu:0`，请按实际机器调整。

### 正确性与边界测试

快速正确性测试：

```bash
python3 ut_ops/test_fused_li_manage.py \
  --device npu:0 \
  --heads 32 \
  --seq-lens 65536 \
  --batch-size 1 \
  --cache-tokens 6144 \
  --miss-count 300 \
  --warmup 1 \
  --iters 3 \
  --seed 7
```

运行默认的 18-bit/21-bit 边界用例：

```bash
python3 ut_ops/test_fused_li_manage.py --device npu:0
```

成功标志：

```text
FUSED_LI_MANAGE_UT_OK
```

### 性能测试

单组性能测试：

```bash
python3 ut_ops/test_fused_li_manage_perf.py \
  --device npu:0 \
  --heads 32 \
  --batch-sizes 24 \
  --seq-lens 20992 \
  --cache-tokens 6144 \
  --miss-ranges 200:200 \
  --warmup 10 \
  --iters 100 \
  --seed 7
```

扫描多组 batch、序列长度和 miss 范围：

```bash
python3 ut_ops/test_fused_li_manage_perf.py \
  --device npu:0 \
  --heads 32 \
  --batch-sizes 1,8,16,24,32,48 \
  --seq-lens 65536,131072 \
  --cache-tokens 8192 \
  --miss-ranges 0:0,0:200,100:300 \
  --warmup 10 \
  --iters 100 \
  --seed 7
```

性能脚本输出 LightningIndexer、完整 `fused_li_manage` 和二者差值，即索引缓存管理增量。成功标志：

```text
FUSED_LI_MANAGE_PERF_UT_OK
```

查看全部参数：

```bash
python3 ut_ops/test_fused_li_manage.py --help
python3 ut_ops/test_fused_li_manage_perf.py --help
```

## MTP4 LI TopK 基线算子测试

`fused_li_manage_mtp` 当前阶段实现 MTP3 的 4 路 Lightning Indexer 分数计算、每路 Top-2048
以及 TopK token 的 HBM hit/miss 判断。四路 query 使用同一个请求的 prefill 满块候选范围。
`topk_src_ids` 当前保留完整 TopK source ID；`topk_dst_slots` 对 miss 写 `-1`，对 hit 写入
合法 HBM slot（包括 slot 0）。TopK 随后按“miss 在前、hit 在后，组内 source ID 升序”
重排。算子使用一次四路 MrgSort 合并有序 miss 前缀，再以单指针扫描去重，写出升序的
`miss_src_ids` 和
`miss_counts`。随后复用非 MTP 的 512-token Sort32/MrgSort eviction 扫描，根据四路
TopK 阈值的交集选择最多 2048 个淘汰候选，并将候选对应的 HBM slot 写入
`miss_dst_slots`。当前阶段仍不修改 cache 映射；union miss 超过 2048 或候选不足时，
对应的 `miss_dst_slots` 前缀写 `-1`。

测试脚本会完成以下检查：

- 将四路 TopK 集合与 `torch_npu.npu_lightning_indexer` 的 MTP4 输出比较；
- 验证 eviction slot 唯一、对应的 token 不属于任一路 TopK，并且 `cache_slots_pool` 没有被修改；
- 使用 NPU Event 分别统计标准 LightningIndexer 和 `fused_li_manage_mtp` 的平均/P50 时延；
- 检查两者的平均时延比，默认上限为 `1.5`。

构建完成后执行默认测试：

```bash
python3 ut_ops/test_fused_li_manage_mtp.py \
  --device npu:0 \
  --batch-size 1 \
  --seq-len 65536 \
  --cache-tokens 8192 \
  --dtype bf16 \
  --warmup 10 \
  --iters 50 \
  --seed 7
```

调整允许的最大平均时延比：

```bash
python3 ut_ops/test_fused_li_manage_mtp.py \
  --device npu:0 \
  --batch-size 1 \
  --seq-len 65536 \
  --cache-tokens 8192 \
  --max-latency-ratio 1.5
```

测试成功标志：

```text
FUSED_LI_MANAGE_MTP_UT_OK
```

查看全部参数：

```bash
python3 ut_ops/test_fused_li_manage_mtp.py --help
```

### MTP3 union 目标负载

#### 当前阶段边界测试

性能测试前，脚本默认执行一组小规模、仅检查正确性的边界用例，覆盖当前算子已经实现的
LI TopK、hit/miss 重排和 unique miss union：

- BF16 与 FP16；
- 最小合法 `source_len=2048`；
- 每路 0 miss、1 miss、64 miss，以及每路全部 2048 个 TopK 均 miss；
- 四路 query 几乎相同和完全不相交两种 union 形态，覆盖 union 恰好达到 8192 容量上限；
- `cache_tokens` 等于和小于 source 长度的两种预算；
- 合法 HBM slot 0、随机物理 block 顺序和非连续 request-pool 行号；
- 动态修改 query 内容后的 ACLGraph replay/eager 一致性；
- 每个用例均检查 TopK、slot、重排顺序、union ID、`miss_counts` 和 cache 不变性。

当前阶段尚未实现的 cache 映射更新以及 union slot 向四路 miss 的回填不在
边界测试范围内。只运行性能回归时可传入 `--skip-boundary-tests` 跳过上述矩阵。

当前测试覆盖四路 LI TopK、hit/miss 重排、unique miss union、最多 2048 个 eviction slot
输出和 `cache_slots_pool` 保持不变。测试不会验证 cache 映射更新或四路 miss slot 回填。

性能数据使用四路相关 query，并通过 `--perf-query-miss-count` 精确控制每一路 TopK 的 miss 数量。
`--perf-query-noise` 控制四路 TopK 的重合程度；默认 `0.25` 时要求每个请求的 TopK union 大约为
3000～4000。构造 miss 时优先选择四路共享和两两共享 token，因此 unique union miss 通常为每路
miss 数量的 1.5～2 倍。`--graph-replays` 额外检查静态输入下 eager 与 ACLGraph replay 输出一致。

目标用例一（每路 200 miss）：

```bash
python3 ut_ops/test_fused_li_manage_mtp.py \
  --device npu:0 \
  --batch-size 24 \
  --source-len 20992 \
  --cache-tokens 8192 \
  --perf-query-miss-count 200 \
  --perf-query-noise 0.25 \
  --graph-replays 3 \
  --warmup 10 \
  --iters 100 \
  --seed 7
```

目标用例二（每路 500 miss）：

```bash
python3 ut_ops/test_fused_li_manage_mtp.py \
  --device npu:0 \
  --batch-size 12 \
  --source-len 40064 \
  --cache-tokens 8192 \
  --perf-query-miss-count 500 \
  --perf-query-noise 0.25 \
  --graph-replays 3 \
  --warmup 10 \
  --iters 100 \
  --seed 7
```

该基线对比使用标准 LightningIndexer 的 `sparse_mode=0`，保证 4 路 query 使用相同的完整 prefill
候选范围，与 `num_candidate_tokens[B]` 的接口语义一致。
