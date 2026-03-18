from __future__ import annotations

import os
import sys


HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
os.environ.setdefault("OBELIX_WEIGHTS", os.path.join(HERE, "weights_nowall.pth"))
os.environ.setdefault("OBELIX_STOCHASTIC", "1")

from agent_rec_ppo import policy  # noqa: E402,F401
