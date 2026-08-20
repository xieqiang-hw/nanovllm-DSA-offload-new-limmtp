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
合法 HBM slot（包括 slot 0）。当前阶段保持 TopK 分数顺序，暂不执行 miss/hit 重排、
四路 miss 并集、淘汰或 cache 映射更新；
`miss_src_ids`、`miss_dst_slots` 和 `miss_counts` 仍保留给后续阶段。

测试脚本会完成以下检查：

- 将四路 TopK 集合与 `torch_npu.npu_lightning_indexer` 的 MTP4 输出比较；
- 验证 `cache_slots_pool` 没有被修改；
- 使用 NPU Event 分别统计标准 LightningIndexer 和 `fused_li_manage_mtp` 的平均/P50 时延；
- 检查两者的平均时延比，默认上限为 `1.25`。

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
  --max-latency-ratio 1.25
```

测试成功标志：

```text
FUSED_LI_MANAGE_MTP_UT_OK
```

查看全部参数：

```bash
python3 ut_ops/test_fused_li_manage_mtp.py --help
```

该基线对比使用标准 LightningIndexer 的 `sparse_mode=0`，保证 4 路 query 使用相同的完整 prefill
候选范围，与 `num_candidate_tokens[B]` 的接口语义一致。
