"""Multiprocess vectorized wrapper for OBELIX."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Iterable, Optional

import numpy as np


def _import_obelix(obelix_py: str):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import OBELIX from {obelix_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def _worker(remote: Connection, obelix_py: str, env_kwargs: dict, seed: int) -> None:
    try:
        OBELIX = _import_obelix(obelix_py)
        env = OBELIX(**env_kwargs, seed=seed)

        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                obs = env.reset(seed=payload)
                remote.send(obs.astype(np.float32, copy=False))
            elif cmd == "step":
                obs, reward, done = env.step(payload, render=False)
                remote.send(
                    (obs.astype(np.float32, copy=False), float(reward), bool(done))
                )
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        pass


class ParallelOBELIX:
    """A simple vectorized OBELIX environment over subprocess workers."""

    def __init__(
        self,
        obelix_py: str,
        num_envs: int,
        base_seed: int,
        env_kwargs: dict,
        mp_start_method: str = "spawn",
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")

        self.num_envs = int(num_envs)
        self._ctx = mp.get_context(mp_start_method)
        self._remotes: list[Connection] = []
        self._processes: list[mp.Process] = []

        for i in range(self.num_envs):
            parent, child = self._ctx.Pipe()
            seed = int(base_seed + i)
            p = self._ctx.Process(
                target=_worker,
                args=(child, obelix_py, env_kwargs, seed),
                daemon=True,
            )
            p.start()
            child.close()
            self._remotes.append(parent)
            self._processes.append(p)

    def reset(self, env_indices: Optional[Iterable[int]] = None, seed: Optional[int] = None):
        if env_indices is None:
            env_indices = range(self.num_envs)
        env_indices = list(env_indices)

        for idx in env_indices:
            worker_seed = None if seed is None else int(seed + idx)
            self._remotes[idx].send(("reset", worker_seed))

        obs_map: dict[int, np.ndarray] = {}
        for idx in env_indices:
            obs_map[idx] = self._remotes[idx].recv()

        return obs_map

    def reset_all(self, seed: Optional[int] = None) -> np.ndarray:
        obs_map = self.reset(env_indices=range(self.num_envs), seed=seed)
        return np.stack([obs_map[i] for i in range(self.num_envs)], axis=0)

    def step(self, actions: list[str]):
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got {len(actions)}"
            )

        for idx, act in enumerate(actions):
            self._remotes[idx].send(("step", act))

        obs = np.empty((self.num_envs, 18), dtype=np.float32)
        rewards = np.empty((self.num_envs,), dtype=np.float32)
        dones = np.empty((self.num_envs,), dtype=bool)

        for idx in range(self.num_envs):
            o, r, d = self._remotes[idx].recv()
            obs[idx] = o
            rewards[idx] = r
            dones[idx] = d

        return obs, rewards, dones

    def close(self) -> None:
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass

        for p in self._processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.kill()
