"""Import selected LakonLab reference modules without its heavy deps (mmcv, CUDA ops, runner)."""

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

LAKONLAB = Path("/Volumes/1T ssd/pi-h3-mlx/LakonLab/lakonlab")


def _swiglu_linear(linear, x, gate_first=True):
    a, b = x.chunk(2, dim=-1)
    gate, up = (a, b) if gate_first else (b, a)
    return linear(F.silu(gate) * up)


class _Registry:
    def register_module(self, *a, **k):
        return lambda cls: cls


def _ensure_stub_packages():
    if "lakonlab" in sys.modules and getattr(sys.modules["lakonlab"], "_pih3_stub", False):
        return
    def pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        return m
    root = pkg("lakonlab")
    root._pih3_stub = True
    ops = pkg("lakonlab.ops")
    ops.swiglu_linear = _swiglu_linear
    pkg("lakonlab.runner")
    ckpt = pkg("lakonlab.runner.checkpoint")
    ckpt._load_cached_checkpoint = ckpt.load_full_state_dict = None
    utils = pkg("lakonlab.utils")
    utils.get_root_logger = lambda: None
    pkg("lakonlab.models")
    builder = pkg("lakonlab.models.builder")
    builder.MODULES = _Registry()
    pkg("lakonlab.models.architectures")
    pkg("lakonlab.models.architectures.sol_h3")


def load_reference(dotted: str, relpath: str):
    _ensure_stub_packages()
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, LAKONLAB / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


def sol_h3_head_cls():
    load_reference("lakonlab.models.architectures.sol_h3.model", "models/architectures/sol_h3/model.py")
    return load_reference("lakonlab.models.architectures.sol_h3.sol_h3",
                          "models/architectures/sol_h3/sol_h3.py").SolH3DWDHead


def randomize_(module: torch.nn.Module, seed: int = 0, std: float = 0.02):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif "scale" in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(std * torch.randn(p.shape, generator=g) * (3 if p.ndim == 1 else 1))
