"""Adversarial Red-vs-Blue helpers for fraud data difficulty control."""

import random
from typing import Any, Dict, List, Tuple

from .data_generator import generate_fraud_data


class FraudDifficultyController:
    """Controls red-team fraud subtlety and tracks blue-team adaptation."""

    LEVELS = {
        1: {"amount_noise": 0.05, "name": "obvious"},
        2: {"amount_noise": 0.15, "name": "moderate"},
        3: {"amount_noise": 0.30, "name": "subtle"},
        4: {"amount_noise": 0.50, "name": "expert"},
        5: {"amount_noise": 0.75, "name": "adversarial"},
    }

    def __init__(self) -> None:
        self.red_level: int = 1
        self.blue_detection_history: List[Dict[str, Any]] = []
        self.arms_race_log: List[Dict[str, Any]] = []

    def generate_adversarial_data(
        self,
        seed: int = 42,
        difficulty: int = 1,
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Generate fraud data with deterministic noise based on difficulty."""
        level = difficulty if difficulty in self.LEVELS else self.red_level
        params = self.LEVELS[level]

        docs, gt = generate_fraud_data(seed)
        rng = random.Random(seed + (level * 101))

        for txn in docs.get("transactions", []):
            txn_id = str(txn.get("txn_id", ""))
            if not txn_id.startswith("TXN-F"):
                continue
            amount = float(txn.get("amount", 0) or 0)
            noise = rng.uniform(-params["amount_noise"], params["amount_noise"])
            new_amount = round(amount * (1 + noise), 2)
            txn["amount"] = max(0.0, new_amount)
            gst_amount = float(txn.get("gst_amount", 0) or 0)
            txn["total"] = round(txn["amount"] + gst_amount, 2)

        return docs, gt

    def record_blue_result(
        self,
        seed: int,
        f1_score: float,
        caught: List[str],
        missed: List[str],
    ) -> None:
        self.blue_detection_history.append(
            {
                "seed": seed,
                "f1": float(f1_score),
                "caught": list(caught),
                "missed": list(missed),
                "red_level": self.red_level,
            }
        )

    def adapt_difficulty(self) -> Dict[str, Any]:
        """Adjust difficulty based on recent blue-team performance."""
        if len(self.blue_detection_history) < 3:
            out = {
                "action": "no_change",
                "old_level": self.red_level,
                "new_level": self.red_level,
                "reason": "insufficient_data",
            }
            self.arms_race_log.append(out)
            return out

        recent = self.blue_detection_history[-5:]
        avg_f1 = sum(float(x["f1"]) for x in recent) / len(recent)
        old = self.red_level

        if avg_f1 > 0.70 and self.red_level < 5:
            self.red_level += 1
            action = "increased"
        elif avg_f1 < 0.30 and self.red_level > 1:
            self.red_level -= 1
            action = "decreased"
        else:
            action = "maintained"

        out = {
            "action": action,
            "old_level": old,
            "new_level": self.red_level,
            "avg_f1": round(avg_f1, 4),
            "samples": len(recent),
        }
        self.arms_race_log.append(out)
        return out

    def get_arms_race_data(self) -> Dict[str, Any]:
        return {
            "current_red_level": self.red_level,
            "level_name": self.LEVELS[self.red_level]["name"],
            "detection_history": self.blue_detection_history,
            "arms_race_log": self.arms_race_log,
        }
