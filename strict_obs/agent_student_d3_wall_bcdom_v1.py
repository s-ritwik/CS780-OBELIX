from __future__ import annotations

import importlib.util
import os
from types import ModuleType


_BASE_AGENT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "PPO",
    "agent.py",
)
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "student_d3_wall_bcdom_v1.pth")


def _load_base_agent() -> ModuleType:
    os.environ["OBELIX_WEIGHTS"] = _WEIGHTS_PATH
    spec = importlib.util.spec_from_file_location("strict_obs_base_agent", _BASE_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base agent from {_BASE_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_AGENT = _load_base_agent()
policy = _BASE_AGENT.policy
