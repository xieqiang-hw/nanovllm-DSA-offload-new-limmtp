# fused_li_manage 算子

本仓库包含 Ascend `fused_li_manage` 自定义算子的实现、构建代码和测试。

- `ut_ops/test_fused_li_manage.py`：正确性、缓存管理语义及 18-bit/21-bit 边界测试。
- `ut_ops/test_fused_li_manage_perf.py`：多 batch、序列长度和 miss 范围的性能测试。

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

主要产物：

- `nanovllm/_C*.so`：Torch 算子注册扩展。
- `nanovllm/_cann_ops_custom/`：本地 CANN 自定义算子包。

## 测试

测试必须在 Ascend NPU 上运行。请先按实际环境设置 CANN、`PYTHONPATH` 和 NPU 设备变量。

正确性及边界测试：

```bash
python3 ut_ops/test_fused_li_manage.py --device npu:0
```

成功标志为 `FUSED_LI_MANAGE_UT_OK`。

性能测试示例：

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

成功标志为 `FUSED_LI_MANAGE_PERF_UT_OK`。

查看全部参数：

```bash
python3 ut_ops/test_fused_li_manage.py --help
python3 ut_ops/test_fused_li_manage_perf.py --help
```
