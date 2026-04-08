from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw = torch.load(args.bundle, map_location="cpu")
    if not isinstance(raw, dict) or "wall" not in raw:
        raise RuntimeError("Expected bundle dict with a 'wall' key")
    torch.save(raw["wall"], args.out)
    print(args.out)


if __name__ == "__main__":
    main()
