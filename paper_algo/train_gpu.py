from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    train_script = base_dir / "train_paper_algo.py"
    default_out = base_dir / "paper_policy.json"

    cmd = [
        sys.executable,
        str(train_script),
        "--env_device",
        "auto",
        "--num_envs",
        "256",
        "--total_env_steps",
        "2000000",
        "--log_interval",
        "50000",
        "--out",
        str(default_out),
    ]
    cmd.extend(sys.argv[1:])

    raise SystemExit(subprocess.call(cmd, cwd=str(base_dir.parent)))


if __name__ == "__main__":
    main()

