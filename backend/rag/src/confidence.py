from typing import Dict, List
import numpy as np


class ConfidenceScorer:

    def compute(
        self,
        validation: Dict,
        rerank_scores: List[float]
    ) -> Dict:

        grounding = validation["grounding_score"]

        rerank_mean = (
            float(np.mean(rerank_scores))
            if rerank_scores else 0.0
        )

        citation_bonus = (
            0.03 if validation["citation_present"]
            else -0.01
        )

        issue_penalty = (
            len(validation["issues"]) * 0.03
        )

        risk_penalty = min(validation["risk_score"], 6) * 0.02

        confidence = (
            grounding * 0.6
            + rerank_mean * 0.3
            + citation_bonus
            - issue_penalty
            - risk_penalty
        )

        confidence = max(0.0, min(1.0, confidence))

        if confidence >= 0.78:
            level = "high"

        elif confidence >= 0.58:
            level = "medium"

        else:
            level = "low"

        return {
            "score": round(confidence, 3),
            "level": level
        }


confidence_scorer = ConfidenceScorer()
