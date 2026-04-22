"""Self-improvement engine with strict seed separation and regression gates."""

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List

from .campaign import CampaignController


TRAIN_SEEDS = list(range(42, 52))
HELD_OUT_SEEDS = list(range(100, 105))


@dataclass
class SelfImproveHistory:
    iterations: List[Dict[str, Any]] = field(default_factory=list)


class SelfImproveEngine:
    """Runs deterministic self-improvement iterations for campaign policy updates."""

    def __init__(self) -> None:
        self._history: Dict[str, SelfImproveHistory] = {}

    def run_iteration(
        self,
        campaign_id: str,
        train_seeds: List[int],
        held_out_seeds: List[int],
        campaign_controller: CampaignController,
    ) -> Dict[str, Any]:
        self._validate_seed_split(train_seeds, held_out_seeds)

        train_baseline = self._pseudo_benchmark(campaign_controller, train_seeds, mode="baseline")
        train_candidate = self._pseudo_benchmark(campaign_controller, train_seeds, mode="candidate")
        learned_delta = train_candidate["mean_score"] - train_baseline["mean_score"]

        baseline = self._pseudo_benchmark(campaign_controller, held_out_seeds, mode="baseline")
        candidate = self._pseudo_benchmark(
            campaign_controller,
            held_out_seeds,
            mode="candidate",
            learned_delta=learned_delta,
        )

        candidate["safety_regression"] = any(
            (cand + 0.02) < base
            for cand, base in zip(candidate["scores"], baseline["scores"])
        )

        transfer_delta = round(candidate["mean_score"] - baseline["mean_score"], 4)
        accepted = (
            learned_delta > 0.005
            and transfer_delta > -0.002
            and not candidate["safety_regression"]
        )

        h = self._history.setdefault(campaign_id, SelfImproveHistory())
        record = {
            "iteration": len(h.iterations) + 1,
            "train_seeds": list(train_seeds),
            "held_out_seeds": list(held_out_seeds),
            "baseline": baseline,
            "candidate": candidate,
            "accepted": accepted,
            "delta": transfer_delta,
            "train_delta": round(learned_delta, 4),
            "transfer_delta": transfer_delta,
        }
        h.iterations.append(record)
        return record

    def get_history(self, campaign_id: str) -> List[Dict[str, Any]]:
        return self._history.get(campaign_id, SelfImproveHistory()).iterations

    @staticmethod
    def _validate_seed_split(train_seeds: List[int], held_out_seeds: List[int]) -> None:
        if not train_seeds or not held_out_seeds:
            raise ValueError("train_seeds and held_out_seeds must be non-empty")
        if set(train_seeds).intersection(set(held_out_seeds)):
            raise ValueError("train_seeds and held_out_seeds must be disjoint")

    @staticmethod
    def _pseudo_benchmark(
        campaign_controller: CampaignController,
        seeds: List[int],
        mode: str,
        learned_delta: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Deterministic placeholder benchmark.

        We avoid nondeterministic model calls at this layer and produce a stable,
        reproducible score trace per seed. This allows API-level integration now,
        while full policy optimization runs in dedicated training scripts.
        """
        campaign_period = campaign_controller.state.current_period if campaign_controller.state.campaign_id else 1

        scores = []
        structural = ((sum(seeds) % 7) - 3) / 1000.0
        transfer = (learned_delta * 0.6) if mode == "candidate" else 0.0
        candidate_adj = transfer + structural if mode == "candidate" else 0.0
        for s in seeds:
            base = ((s % 13) / 100.0) + 0.45 + (campaign_period * 0.002)
            adj = candidate_adj
            scores.append(min(0.99, max(0.01, round(base + adj, 4))))

        mean_score = round(mean(scores), 4)
        safety_regression = False

        return {
            "scores": scores,
            "mean_score": mean_score,
            "safety_regression": safety_regression,
        }
