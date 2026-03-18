from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
BEHAVIORS = ("find", "push", "unwedge")
BASE_OBS_DIM = 18
POLICY_STATE_DIM = 19
PREV_STUCK_MEMORY_INDEX = 18
FORWARD_ACTION = ACTIONS.index("FW")
L45_ACTION = ACTIONS.index("L45")
R45_ACTION = ACTIONS.index("R45")
TURN_LEFT_ACTIONS = frozenset((ACTIONS.index("L45"), ACTIONS.index("L22")))
TURN_RIGHT_ACTIONS = frozenset((ACTIONS.index("R22"), ACTIONS.index("R45")))

# Observation groups.
LEFT_BITS = np.array([0, 1, 2, 3], dtype=np.int64)
FORWARD_BITS = np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
RIGHT_BITS = np.array([12, 13, 14, 15], dtype=np.int64)
# Central front-facing sonar bits.
FINDER_MATCH_BITS = np.array([6, 8], dtype=np.int64)
FINDER_FRONT_PROGRESS_BITS = np.array([5, 6, 7, 8, 9, 10], dtype=np.int64)
BUMP_BIT = 16
STUCK_BIT = 17


@dataclass(frozen=True)
class IntrinsicRewardConfig:
    find_forward_match_reward: float = 3.0
    find_forward_progress_reward: float = 1.0
    find_forward_explore_reward: float = 0.10
    find_forward_blind_penalty: float = -0.35
    find_turn_toward_bonus: float = 0.75
    find_turn_away_penalty: float = -0.75
    find_turn_idle_penalty: float = -0.05
    find_turn_acquire_bonus: float = 0.50
    find_recenter_bonus: float = 1.00
    find_post_stuck_forward_penalty: float = -0.90
    find_post_stuck_turn_bonus: float = 0.35
    push_forward_reward: float = 1.0
    push_contact_loss_penalty: float = -3.0
    unwedge_clear_reward: float = 1.0
    unwedge_stuck_penalty: float = -3.0


def binarize_observation(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


def encode_policy_state(obs_bits: np.ndarray, prev_stuck_bit: int | bool) -> np.ndarray:
    state_bits = np.zeros((POLICY_STATE_DIM,), dtype=np.int8)
    state_bits[:BASE_OBS_DIM] = np.asarray(obs_bits[:BASE_OBS_DIM], dtype=np.int8)
    state_bits[PREV_STUCK_MEMORY_INDEX] = 1 if bool(prev_stuck_bit) else 0
    return state_bits


def previous_stuck_memory(state_bits: np.ndarray) -> bool:
    return bool(state_bits[PREV_STUCK_MEMORY_INDEX])


def finder_match(obs_bits: np.ndarray) -> bool:
    return bool(np.any(obs_bits[FINDER_MATCH_BITS] > 0))


def any_visible(obs_bits: np.ndarray) -> bool:
    return bool(np.any(obs_bits[:16] > 0))


def left_visible(obs_bits: np.ndarray) -> bool:
    return bool(np.any(obs_bits[LEFT_BITS] > 0))


def right_visible(obs_bits: np.ndarray) -> bool:
    return bool(np.any(obs_bits[RIGHT_BITS] > 0))


def forward_visible(obs_bits: np.ndarray) -> bool:
    return bool(np.any(obs_bits[FORWARD_BITS] > 0))


def forward_progress_visible(obs_bits: np.ndarray) -> bool:
    return bool(np.any(obs_bits[FINDER_FRONT_PROGRESS_BITS] > 0))


def update_behavior_timers(
    obs_bits: np.ndarray,
    push_timer: int,
    unwedge_timer: int,
    push_persistence_steps: int,
    unwedge_persistence_steps: int,
) -> tuple[int, int]:
    if bool(obs_bits[BUMP_BIT]):
        push_timer = int(push_persistence_steps)
    elif push_timer > 0:
        push_timer -= 1

    if bool(obs_bits[STUCK_BIT]):
        unwedge_timer = int(unwedge_persistence_steps)
    elif unwedge_timer > 0:
        unwedge_timer -= 1

    return max(0, int(push_timer)), max(0, int(unwedge_timer))


def select_behavior(obs_bits: np.ndarray, push_timer: int, unwedge_timer: int) -> str:
    if bool(obs_bits[STUCK_BIT]) or unwedge_timer > 0:
        return "unwedge"
    if bool(obs_bits[BUMP_BIT]) or push_timer > 0:
        return "push"
    return "find"


def behavior_reward(
    behavior: str,
    obs_bits: np.ndarray,
    action_idx: int,
    next_obs_bits: np.ndarray,
    prev_stuck_bit: int | bool = False,
    config: IntrinsicRewardConfig | None = None,
) -> float:
    if config is None:
        config = IntrinsicRewardConfig()

    if behavior == "find":
        prev_stuck = bool(prev_stuck_bit)
        current_any = any_visible(obs_bits)
        current_left = left_visible(obs_bits)
        current_right = right_visible(obs_bits)
        current_forward = forward_visible(obs_bits)
        next_any = any_visible(next_obs_bits)
        next_forward = forward_visible(next_obs_bits)

        if action_idx == FORWARD_ACTION:
            if finder_match(next_obs_bits):
                return config.find_forward_match_reward
            if forward_progress_visible(next_obs_bits):
                return config.find_forward_progress_reward
            if (not current_any) and (not next_any):
                reward = config.find_forward_explore_reward
            else:
                reward = config.find_forward_blind_penalty
            if prev_stuck:
                reward += config.find_post_stuck_forward_penalty
            return reward

        reward = 0.0
        turned_left = action_idx in TURN_LEFT_ACTIONS
        turned_right = action_idx in TURN_RIGHT_ACTIONS

        if current_left and not current_right:
            reward += config.find_turn_toward_bonus if turned_left else config.find_turn_away_penalty
        elif current_right and not current_left:
            reward += config.find_turn_toward_bonus if turned_right else config.find_turn_away_penalty
        elif not current_any:
            reward += config.find_turn_idle_penalty

        if prev_stuck and (not current_forward):
            reward += config.find_post_stuck_turn_bonus

        if finder_match(next_obs_bits):
            reward += config.find_recenter_bonus
        elif (not current_forward) and next_forward:
            reward += config.find_turn_acquire_bonus

        return reward

    if behavior == "push":
        had_contact = bool(obs_bits[BUMP_BIT])
        has_contact = bool(next_obs_bits[BUMP_BIT])
        is_stuck = bool(next_obs_bits[STUCK_BIT])
        if action_idx == FORWARD_ACTION and has_contact and not is_stuck:
            return config.push_forward_reward
        if had_contact and not has_contact:
            return config.push_contact_loss_penalty
        return 0.0

    if behavior == "unwedge":
        if not bool(next_obs_bits[STUCK_BIT]):
            return config.unwedge_clear_reward
        return config.unwedge_stuck_penalty

    raise ValueError(f"Unknown behavior: {behavior}")


@dataclass
class Cluster:
    probs: np.ndarray
    q_value: float
    count: int

    def match_probability(self, state_bits: np.ndarray, eps: float = 1e-6) -> float:
        probs = np.clip(self.probs, eps, 1.0 - eps)
        likelihood_terms = np.where(state_bits > 0, probs, 1.0 - probs)
        return float(np.exp(np.sum(np.log(likelihood_terms.astype(np.float64)))))

    def merge_state(self, state_bits: np.ndarray, q_value: float) -> None:
        state_arr = np.asarray(state_bits, dtype=np.float32)
        total = float(self.count + 1)
        self.probs = ((self.probs * self.count) + state_arr) / total
        self.q_value = ((self.q_value * self.count) + float(q_value)) / total
        self.count += 1

    def merge_cluster(self, other: "Cluster") -> None:
        total = float(self.count + other.count)
        self.probs = ((self.probs * self.count) + (other.probs * other.count)) / total
        self.q_value = ((self.q_value * self.count) + (other.q_value * other.count)) / total
        self.count += other.count

    def to_dict(self) -> dict[str, Any]:
        return {
            "probs": self.probs.astype(float).tolist(),
            "q_value": float(self.q_value),
            "count": int(self.count),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Cluster":
        return cls(
            probs=np.asarray(payload["probs"], dtype=np.float32),
            q_value=float(payload["q_value"]),
            count=int(payload["count"]),
        )


class ClusteredActionModel:
    def __init__(
        self,
        obs_dim: int,
        match_threshold: float,
        q_match_delta: float,
        cluster_distance_threshold: float,
        max_clusters: int,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.match_threshold = float(match_threshold)
        self.q_match_delta = float(q_match_delta)
        self.cluster_distance_threshold = float(cluster_distance_threshold)
        self.max_clusters = int(max_clusters)
        self.clusters: list[Cluster] = []

    def estimate(self, state_bits: np.ndarray) -> float:
        if not self.clusters:
            return 0.0

        weights = np.asarray(
            [cluster.match_probability(state_bits) for cluster in self.clusters],
            dtype=np.float64,
        )
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            return 0.0

        q_values = np.asarray([cluster.q_value for cluster in self.clusters], dtype=np.float64)
        return float(np.dot(weights, q_values) / total_weight)

    def _best_matching_cluster(self, state_bits: np.ndarray, q_value: float) -> int | None:
        best_idx = None
        best_prob = self.match_threshold
        for idx, cluster in enumerate(self.clusters):
            match_prob = cluster.match_probability(state_bits)
            if match_prob < self.match_threshold:
                continue
            if abs(cluster.q_value - q_value) > self.q_match_delta:
                continue
            if match_prob > best_prob:
                best_idx = idx
                best_prob = match_prob
        return best_idx

    def _merge_around_index(self, base_idx: int) -> None:
        if len(self.clusters) <= 1 or base_idx < 0 or base_idx >= len(self.clusters):
            return

        changed = True
        while changed and len(self.clusters) > 1:
            changed = False
            base = self.clusters[base_idx]
            for other_idx, other in enumerate(self.clusters):
                if other_idx == base_idx:
                    continue
                if abs(base.q_value - other.q_value) > self.q_match_delta:
                    continue
                distance = float(np.linalg.norm(base.probs - other.probs))
                if distance > self.cluster_distance_threshold:
                    continue
                base.merge_cluster(other)
                del self.clusters[other_idx]
                if other_idx < base_idx:
                    base_idx -= 1
                changed = True
                break

    def _enforce_cluster_limit(self) -> None:
        while len(self.clusters) > self.max_clusters:
            smallest_idx = min(range(len(self.clusters)), key=lambda idx: self.clusters[idx].count)
            if len(self.clusters) == 1:
                break

            base_idx = 0 if smallest_idx != 0 else 1
            best_idx = base_idx
            best_key = (
                float(np.linalg.norm(self.clusters[smallest_idx].probs - self.clusters[base_idx].probs)),
                abs(self.clusters[smallest_idx].q_value - self.clusters[base_idx].q_value),
            )
            for idx in range(len(self.clusters)):
                if idx == smallest_idx:
                    continue
                key = (
                    float(np.linalg.norm(self.clusters[smallest_idx].probs - self.clusters[idx].probs)),
                    abs(self.clusters[smallest_idx].q_value - self.clusters[idx].q_value),
                )
                if key < best_key:
                    best_key = key
                    best_idx = idx

            self.clusters[best_idx].merge_cluster(self.clusters[smallest_idx])
            del self.clusters[smallest_idx]
            if best_idx > smallest_idx:
                best_idx -= 1
            self._merge_around_index(best_idx)

    def update(self, state_bits: np.ndarray, q_value: float) -> None:
        state_arr = np.asarray(state_bits, dtype=np.float32)
        cluster_idx = self._best_matching_cluster(state_arr, q_value)

        if cluster_idx is None:
            self.clusters.append(
                Cluster(probs=state_arr.copy(), q_value=float(q_value), count=1)
            )
            cluster_idx = len(self.clusters) - 1
        else:
            self.clusters[cluster_idx].merge_state(state_arr, q_value)

        self._merge_around_index(cluster_idx)
        self._enforce_cluster_limit()

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "obs_dim": self.obs_dim,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        match_threshold: float,
        q_match_delta: float,
        cluster_distance_threshold: float,
        max_clusters: int,
    ) -> "ClusteredActionModel":
        model = cls(
            obs_dim=int(payload.get("obs_dim", 18)),
            match_threshold=match_threshold,
            q_match_delta=q_match_delta,
            cluster_distance_threshold=cluster_distance_threshold,
            max_clusters=max_clusters,
        )
        model.clusters = [Cluster.from_dict(item) for item in payload.get("clusters", [])]
        return model


class ClusteredBehaviorQ:
    def __init__(
        self,
        obs_dim: int = BASE_OBS_DIM,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        match_threshold: float = 1e-4,
        q_match_delta: float = 1.5,
        cluster_distance_threshold: float = 1.0,
        max_clusters_per_action: int = 256,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.match_threshold = float(match_threshold)
        self.q_match_delta = float(q_match_delta)
        self.cluster_distance_threshold = float(cluster_distance_threshold)
        self.max_clusters_per_action = int(max_clusters_per_action)
        self.actions = [
            ClusteredActionModel(
                obs_dim=self.obs_dim,
                match_threshold=self.match_threshold,
                q_match_delta=self.q_match_delta,
                cluster_distance_threshold=self.cluster_distance_threshold,
                max_clusters=self.max_clusters_per_action,
            )
            for _ in ACTIONS
        ]

    def q_values(self, state_bits: np.ndarray) -> np.ndarray:
        return np.asarray(
            [action_model.estimate(state_bits) for action_model in self.actions],
            dtype=np.float32,
        )

    def greedy_action(self, state_bits: np.ndarray, rng: np.random.Generator) -> int:
        q_values = self.q_values(state_bits)
        max_q = float(np.max(q_values))
        candidates = np.flatnonzero(np.isclose(q_values, max_q))
        return int(rng.choice(candidates))

    def select_action(self, state_bits: np.ndarray, rng: np.random.Generator) -> int:
        if float(rng.random()) < self.epsilon:
            return int(rng.integers(0, len(ACTIONS)))
        return self.greedy_action(state_bits, rng)

    def update(
        self,
        state_bits: np.ndarray,
        action_idx: int,
        reward: float,
        next_state_bits: np.ndarray,
        next_applicable: bool,
    ) -> float:
        current_q = float(self.actions[action_idx].estimate(state_bits))
        next_value = 0.0
        if next_applicable:
            next_value = float(np.max(self.q_values(next_state_bits)))
        target = float(reward) + self.gamma * next_value
        updated_q = current_q + self.alpha * (target - current_q)
        self.actions[action_idx].update(state_bits, updated_q)
        return updated_q

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "match_threshold": self.match_threshold,
            "q_match_delta": self.q_match_delta,
            "cluster_distance_threshold": self.cluster_distance_threshold,
            "max_clusters_per_action": self.max_clusters_per_action,
            "actions": [action_model.to_dict() for action_model in self.actions],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClusteredBehaviorQ":
        model = cls(
            obs_dim=int(payload.get("obs_dim", 18)),
            alpha=float(payload.get("alpha", 0.3)),
            gamma=float(payload.get("gamma", 0.95)),
            epsilon=float(payload.get("epsilon", 0.1)),
            match_threshold=float(payload.get("match_threshold", 1e-4)),
            q_match_delta=float(payload.get("q_match_delta", 1.5)),
            cluster_distance_threshold=float(payload.get("cluster_distance_threshold", 1.0)),
            max_clusters_per_action=int(payload.get("max_clusters_per_action", 256)),
        )
        model.actions = [
            ClusteredActionModel.from_dict(
                item,
                match_threshold=model.match_threshold,
                q_match_delta=model.q_match_delta,
                cluster_distance_threshold=model.cluster_distance_threshold,
                max_clusters=model.max_clusters_per_action,
            )
            for item in payload.get("actions", [])
        ]
        return model


class PaperSubsumptionController:
    def __init__(
        self,
        obs_dim: int = BASE_OBS_DIM,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        match_threshold: float = 1e-4,
        q_match_delta: float = 1.5,
        cluster_distance_threshold: float = 1.0,
        max_clusters_per_action: int = 256,
        push_persistence_steps: int = 5,
        unwedge_persistence_steps: int = 5,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.push_persistence_steps = int(push_persistence_steps)
        self.unwedge_persistence_steps = int(unwedge_persistence_steps)
        self.modules = {
            behavior: ClusteredBehaviorQ(
                obs_dim=self.obs_dim,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                match_threshold=match_threshold,
                q_match_delta=q_match_delta,
                cluster_distance_threshold=cluster_distance_threshold,
                max_clusters_per_action=max_clusters_per_action,
            )
            for behavior in BEHAVIORS
        }

    def behavior_for_state(
        self,
        obs_bits: np.ndarray,
        push_timer: int,
        unwedge_timer: int,
    ) -> tuple[str, int, int]:
        push_timer, unwedge_timer = update_behavior_timers(
            obs_bits=obs_bits,
            push_timer=push_timer,
            unwedge_timer=unwedge_timer,
            push_persistence_steps=self.push_persistence_steps,
            unwedge_persistence_steps=self.unwedge_persistence_steps,
        )
        behavior = select_behavior(obs_bits, push_timer, unwedge_timer)
        return behavior, push_timer, unwedge_timer

    def select_action(
        self,
        behavior: str,
        obs_bits: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        return self.modules[behavior].select_action(obs_bits, rng)

    def greedy_action(
        self,
        behavior: str,
        obs_bits: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        return self.modules[behavior].greedy_action(obs_bits, rng)

    def save(self, out_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "actions": ACTIONS,
            "behaviors": list(BEHAVIORS),
            "push_persistence_steps": self.push_persistence_steps,
            "unwedge_persistence_steps": self.unwedge_persistence_steps,
            "metadata": metadata or {},
            "modules": {name: module.to_dict() for name, module in self.modules.items()},
        }
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PaperSubsumptionController":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        modules = payload.get("modules", {})
        first_module = next(iter(modules.values()))
        controller = cls(
            obs_dim=int(first_module.get("obs_dim", 18)),
            alpha=float(first_module.get("alpha", 0.3)),
            gamma=float(first_module.get("gamma", 0.95)),
            epsilon=float(first_module.get("epsilon", 0.1)),
            match_threshold=float(first_module.get("match_threshold", 1e-4)),
            q_match_delta=float(first_module.get("q_match_delta", 1.5)),
            cluster_distance_threshold=float(first_module.get("cluster_distance_threshold", 1.0)),
            max_clusters_per_action=int(first_module.get("max_clusters_per_action", 256)),
            push_persistence_steps=int(payload.get("push_persistence_steps", 5)),
            unwedge_persistence_steps=int(payload.get("unwedge_persistence_steps", 5)),
        )
        controller.modules = {
            name: ClusteredBehaviorQ.from_dict(module_payload)
            for name, module_payload in modules.items()
        }
        return controller
