from typing import Dict


class AnswerPolicy:

    def evaluate(
        self,
        validation: Dict,
        confidence: Dict
    ) -> Dict:
        grounding_score = validation["grounding_score"]
        has_contexts = validation.get("has_meaningful_contexts", True)
        catastrophic = validation.get("catastrophic", False)
        severe_unsupported = validation.get("severe_unsupported_references", False)

        if not has_contexts:
            return {
                "accepted": False,
                "reason": "No meaningful retrieved context",
                "response_mode": "reject",
            }

        if grounding_score < 0.45:
            return {
                "accepted": False,
                "reason": "Answer not sufficiently grounded",
                "response_mode": "reject",
            }

        if severe_unsupported:
            return {
                "accepted": False,
                "reason": "Severe unsupported references detected",
                "response_mode": "reject",
            }

        if catastrophic:
            return {
                "accepted": False,
                "reason": "Catastrophic hallucination indicators detected",
                "response_mode": "reject",
            }

        if confidence["score"] >= 0.78 and grounding_score >= 0.7:
            return {
                "accepted": True,
                "reason": "Answer validated",
                "response_mode": "normal",
            }

        if confidence["score"] >= 0.58 or grounding_score >= 0.6:
            return {
                "accepted": True,
                "reason": "Answer validated with caution",
                "response_mode": "cautious",
            }

        return {
            "accepted": True,
            "reason": "Answer validated with limited support",
            "response_mode": "limited",
        }


policy_engine = AnswerPolicy()
