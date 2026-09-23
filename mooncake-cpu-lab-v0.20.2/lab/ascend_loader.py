"""Load rc1's genuine scheduler without starting its EngineCore processes.

The upstream file also replaces process launchers at module scope. Select only
its imports, feature gate and BalanceScheduler AST nodes, without rewriting any
method. This is a control-plane harness boundary, not a torch_npu emulation.
"""
import ast
import sys
from pathlib import Path
from types import ModuleType


def load_scheduler():
    name = 'lab_upstream_ascend_scheduler'
    if name in sys.modules:
        return sys.modules[name].BalanceScheduler
    path = Path(__file__).resolve().parents[1] / 'vendor/vllm_ascend/patch/platform/patch_balance_schedule.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    excluded_imports = {
        'vllm.v1.engine.core',
        'vllm.transformers_utils.config',
        'vllm.utils.system_utils',
    }
    definitions = {'_balance_scheduling_enabled', 'BalanceScheduler'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            selected.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module not in excluded_imports:
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions:
            selected.append(node)
    if {n.name for n in selected if isinstance(n, (ast.FunctionDef, ast.ClassDef))} != definitions:
        raise RuntimeError('Pinned Ascend scheduler structure changed; review the loader')
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module.BalanceScheduler
