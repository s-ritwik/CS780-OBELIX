from __future__ import annotations

import os
import sys


HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
os.environ.setdefault("OBELIX_WEIGHTS", os.path.join(HERE, "eval300_mixed_best.pth"))
os.environ.setdefault("OBELIX_STOCHASTIC", "0")

from agent import policy  # noqa: E402,F401
