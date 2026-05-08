import logging
import os
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Keep CPU pressure low on Mac.
torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class AnswerValidator:
    """Validate generated answers against retrieved legal context."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        grounding_threshold: float = 0.65,
        max_risk_score: int = 4,
    ):
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.grounding_threshold = grounding_threshold
        self.max_risk_score = max_risk_score

        self.risky_patterns = {
            "absolute_claims": [
                r"\balways\b",
                r"\bnever\b",
                r"\ball cases\b",
                r"\bwithout exception\b",
                r"\bguaranteed\b",
                r"\buniversally\b",
                r"\bcannot be granted\b",
                r"\bcannot be denied\b",
                r"\bmust be granted\b",
                r"\bmust be rejected\b",
                r"\bautomatically entitled\b",
                r"\bautomatically rejected\b",
            ],
            "fake_authority": [
                r"\bsettled law\b",
                r"\bwell[- ]established law\b",
                r"\bbinding precedent\b",
                r"\bmandatory principle\b",
                r"\buniformly held\b",
                r"\bconsistently held\b",
                r"\bclearly established\b",
                r"\bthe law is clear\b",
            ],
            "speculative_reasoning": [
                r"\blikely guilty\b",
                r"\bappears guilty\b",
                r"\bprobably\b",
                r"\bmust have intended\b",
                r"\bseems to have\b",
                r"\bpossibly\b",
                r"\bmay have\b",
            ],
            "unsupported_factual_claims": [
                r"\bproved beyond doubt\b",
                r"\bconclusively established\b",
                r"\bdefinitely involved\b",
                r"\bclearly involved\b",
                r"\bcertainly committed\b",
                r"\bobviously guilty\b",
            ],
            "overgeneralization": [
                r"\bfinancial fraud always\b",
                r"\beconomic offences are always\b",
                r"\bbail is never granted\b",
                r"\bcourts uniformly reject\b",
                r"\ball bail applications\b",
                r"\bfraud cases always\b",
            ],
        }

        self.citation_patterns = {
            "court_names": [
                r"\bSupreme Court of India\b",
                r"\bSupreme Court\b",
                r"\b[A-Z][A-Za-z.&' -]+ High Court\b",
                r"\bDistrict Court\b",
                r"\bSessions Court\b",
                r"\bSpecial Court\b",
                r"\bTrial Court\b",
            ],
            "year": [
                r"\b(19|20)\d{2}\b",
                r"\(\d{4}\)",
            ],
            "statutes": [
                r"\bIPC\b(?:\s*[-:]?\s*\d+)?",
                r"\bCrPC\b(?:\s*[-:]?\s*\d+)?",
                r"\bUAPA\b",
                r"\bNDPS\b",
                r"\bPMLA\b",
                r"\bBNSS\b",
                r"\bBNS\b",
                r"\bEvidence Act\b",
            ],
            "legal_reasoning_markers": [
                r"\bprima facie\b",
                r"\bheld that\b",
                r"\bobserved that\b",
                r"\bconsidered that\b",
                r"\brejected bail\b",
                r"\bgranted bail\b",
                r"\bcancelled bail\b",
            ],
        }

        logger.info(
            "AnswerValidator loaded: model=%s device=%s grounding_threshold=%.2f max_risk_score=%d",
            model_name,
            DEVICE,
            grounding_threshold,
            max_risk_score,
        )

    def _normalize(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _compile_hits(self, text: str, patterns: List[str]) -> List[str]:
        hits = []
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(pat)
        return hits

    def _extract_court_mentions(self, text: str) -> Set[str]:
        mentions = set()
        if not text:
            return mentions

        court_regex = re.compile(
            r"\b(?:Supreme Court(?: of India)?|[A-Z][A-Za-z.&' -]+ High Court|District Court|Sessions Court|Special Court|Trial Court)\b",
            flags=re.IGNORECASE,
        )

        for m in court_regex.findall(text):
            mentions.add(self._normalize(m))
        return mentions

    def _extract_statute_mentions(self, text: str) -> Set[str]:
        mentions = set()
        if not text:
            return mentions

        statute_regex = re.compile(
            r"\b(?:IPC|CrPC|UAPA|NDPS|PMLA|BNSS|BNS|Evidence Act)\b(?:\s*[-:]?\s*\d+)?",
            flags=re.IGNORECASE,
        )
        for m in statute_regex.findall(text):
            mentions.add(self._normalize(m))
        return mentions

    def _extract_years(self, text: str) -> Set[str]:
        years = set()
        if not text:
            return years

        for m in re.finditer(r"\b(19|20)\d{2}\b", text):
            years.add(m.group(0))
        return years

    def _extract_context_entities(self, contexts: List[str]) -> Dict[str, Set[str]]:
        combined = "\n".join(contexts or [])

        courts = set()
        statutes = set()
        years = set()

        for line in combined.splitlines():
            line_clean = line.strip()
            if line_clean.lower().startswith("court:"):
                courts.update(self._extract_court_mentions(line_clean))
            elif line_clean.lower().startswith("year:"):
                years.update(re.findall(r"\b(19|20)\d{2}\b", line_clean))
            elif line_clean.lower().startswith("ipc sections:"):
                statutes.update(self._extract_statute_mentions(line_clean))

        courts.update(self._extract_court_mentions(combined))
        statutes.update(self._extract_statute_mentions(combined))
        years.update(self._extract_years(combined))

        return {
            "courts": courts,
            "statutes": statutes,
            "years": years,
        }

    def compute_grounding_score(self, query: str, contexts: List[str], answer: str) -> Dict[str, float]:
        """Compute grounding against the query and retrieved context."""
        if not contexts:
            return {
                "context_score": 0.0,
                "query_score": 0.0,
                "grounding_score": 0.0,
            }

        combined_context = "\n".join(contexts)

        embeddings = self.model.encode(
            [answer, combined_context, query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        answer_vec = embeddings[0]
        context_vec = embeddings[1]
        query_vec = embeddings[2]

        context_score = float(np.dot(answer_vec, context_vec))
        query_score = float(np.dot(answer_vec, query_vec))

        grounding_score = 0.75 * context_score + 0.25 * query_score

        return {
            "context_score": round(context_score, 4),
            "query_score": round(query_score, 4),
            "grounding_score": round(float(grounding_score), 4),
        }

    def detect_risky_language(self, answer: str) -> Tuple[List[str], List[Dict]]:
        """Detect heuristic hallucination signals in the answer."""
        issues: List[str] = []
        details: List[Dict] = []

        for category, patterns in self.risky_patterns.items():
            for pat in patterns:
                if re.search(pat, answer, flags=re.IGNORECASE):
                    msg = f"Risky language detected ({category}): '{pat}'"
                    issues.append(msg)
                    details.append(
                        {
                            "type": "risky_language",
                            "category": category,
                            "pattern": pat,
                            "severity": self._severity_for_category(category),
                            "message": msg,
                        }
                    )

        return issues, details

    def _severity_for_category(self, category: str) -> int:
        severity_map = {
            "absolute_claims": 3,
            "fake_authority": 3,
            "unsupported_factual_claims": 3,
            "overgeneralization": 2,
            "speculative_reasoning": 2,
        }
        return severity_map.get(category, 1)

    def _is_catastrophic_issue(self, detail: Dict) -> bool:
        if detail.get("type") != "risky_language":
            return False
        return detail.get("category") in {
            "fake_authority",
            "unsupported_factual_claims",
        }

    def _has_severe_unsupported_reference(self, issue_details: List[Dict]) -> bool:
        return any(
            detail.get("type") == "unsupported_reference"
            and detail.get("severity", 0) >= 3
            for detail in issue_details
        )

    def validate_citations(self, answer: str) -> Dict:
        """Check whether the answer contains legal-style references."""
        hit_map = {}

        for category, patterns in self.citation_patterns.items():
            hits = self._compile_hits(answer, patterns)
            hit_map[category] = hits

        has_court = len(hit_map["court_names"]) > 0
        has_year = len(hit_map["year"]) > 0
        has_statute = len(hit_map["statutes"]) > 0
        has_reasoning = len(hit_map["legal_reasoning_markers"]) > 0

        citation_present = has_court or has_year or has_statute or has_reasoning

        return {
            "present": citation_present,
            "court_hits": hit_map["court_names"],
            "year_hits": hit_map["year"],
            "statute_hits": hit_map["statutes"],
            "reasoning_hits": hit_map["legal_reasoning_markers"],
        }

    def detect_unsupported_references(self, answer: str, contexts: List[str]) -> List[Dict]:
        """Find references in the answer that are absent from the context."""
        issues: List[Dict] = []
        entities = self._extract_context_entities(contexts)

        answer_courts = self._extract_court_mentions(answer)
        answer_statutes = self._extract_statute_mentions(answer)
        answer_years = self._extract_years(answer)

        if entities["courts"]:
            unsupported_courts = answer_courts - entities["courts"]
            for court in unsupported_courts:
                issues.append(
                    {
                        "type": "unsupported_reference",
                        "category": "court",
                        "value": court,
                        "severity": 3,
                        "message": f"Answer references unsupported court: '{court}'",
                    }
                )

        if entities["statutes"]:
            unsupported_statutes = answer_statutes - entities["statutes"]
            for statute in unsupported_statutes:
                issues.append(
                    {
                        "type": "unsupported_reference",
                        "category": "statute",
                        "value": statute,
                        "severity": 2,
                        "message": f"Answer references unsupported statute: '{statute}'",
                    }
                )

        if entities["years"]:
            unsupported_years = answer_years - entities["years"]
            for year in unsupported_years:
                issues.append(
                    {
                        "type": "unsupported_reference",
                        "category": "year",
                        "value": year,
                        "severity": 1,
                        "message": f"Answer references unsupported year: '{year}'",
                    }
                )

        return issues

    def validate(self, query: str, contexts: List[str], answer: str) -> Dict:
        """Run the validation checks for a generated answer."""
        issues: List[str] = []
        issue_details: List[Dict] = []

        if not answer or not answer.strip():
            return {
                "supported": False,
                "grounding_score": 0.0,
                "context_score": 0.0,
                "query_score": 0.0,
                "citation_present": False,
                "issues": ["Empty answer"],
                "issue_details": [{"type": "empty_answer", "severity": 3, "message": "Empty answer"}],
                "risk_score": 3,
                "risk_level": "high",
                "catastrophic": True,
                "severe_unsupported_references": False,
                "has_meaningful_contexts": False,
            }

        if not contexts:
            return {
                "supported": False,
                "grounding_score": 0.0,
                "context_score": 0.0,
                "query_score": 0.0,
                "citation_present": False,
                "issues": ["No retrieved contexts to validate against"],
                "issue_details": [
                    {
                        "type": "no_context",
                        "severity": 3,
                        "message": "No retrieved contexts to validate against",
                    }
                ],
                "risk_score": 3,
                "risk_level": "high",
                "catastrophic": True,
                "severe_unsupported_references": False,
                "has_meaningful_contexts": False,
            }

        grounding = self.compute_grounding_score(query=query, contexts=contexts, answer=answer)
        risky_issues, risky_details = self.detect_risky_language(answer)
        issues.extend(risky_issues)
        issue_details.extend(risky_details)

        unsupported_details = self.detect_unsupported_references(answer, contexts)
        for d in unsupported_details:
            issues.append(d["message"])
        issue_details.extend(unsupported_details)

        citation_info = self.validate_citations(answer)
        citation_present = citation_info["present"]
        if not citation_present:
            issues.append("Missing citation-like legal reference")
            issue_details.append(
                {
                    "type": "missing_citation",
                    "severity": 2,
                    "message": "Missing citation-like legal reference",
                }
            )

        risk_score = int(sum(d.get("severity", 1) for d in issue_details))

        if risk_score >= 6:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "medium"
        else:
            risk_level = "low"

        catastrophic = any(
            self._is_catastrophic_issue(detail)
            for detail in issue_details
        )
        severe_unsupported_references = self._has_severe_unsupported_reference(issue_details)

        supported = (
            grounding["grounding_score"] >= self.grounding_threshold
            and not catastrophic
            and not severe_unsupported_references
        )

        return {
            "supported": supported,
            "grounding_score": grounding["grounding_score"],
            "context_score": grounding["context_score"],
            "query_score": grounding["query_score"],
            "citation_present": citation_present,
            "citation_info": citation_info,
            "issues": issues,
            "issue_details": issue_details,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "catastrophic": catastrophic,
            "severe_unsupported_references": severe_unsupported_references,
            "has_meaningful_contexts": bool(contexts),
        }


validator = AnswerValidator()
