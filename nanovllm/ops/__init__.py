from __future__ import annotations
import importlib
import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_VENDOR = _PACKAGE_DIR / "_cann_ops_custom" / "vendors" / "nanovllm-ascend"
if _VENDOR.exists():
    old_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(_VENDOR) if not old_path else str(_VENDOR) + os.pathsep + old_path
    opapi = _VENDOR / "op_api" / "lib" / "libcust_opapi.so"
    if opapi.is_file():
        os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(opapi)
importlib.import_module("torch_npu")
torch = importlib.import_module("torch")
importlib.import_module("nanovllm._C")
fused_li_manage = torch.ops.nanovllm_dsa.fused_li_manage.default
fused_li_manage_mtp = torch.ops.nanovllm_dsa.fused_li_manage_mtp.default
__all__ = ["fused_li_manage", "fused_li_manage_mtp"]
