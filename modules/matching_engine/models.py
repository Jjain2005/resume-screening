from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from ..resume_processing.models import SectionType

class MatchClassification(str, Enum):
    EXACT_MATCH = "Exact Match"
    STRONG_SEMANTIC_MATCH = "Strong Semantic Match"
    FUZZY_MATCH = "Fuzzy Match"
    RELATED_SKILL_NOT_EQUIVALENT = "Related Skill – Not Equivalent"
    PARTIAL_EVIDENCE = "Partial Evidence"
    NOT_FOUND = "Not Found"
    NEEDS_MANUAL_REVIEW = "Needs Manual Review"

class EvidenceStrength(str, Enum):
    NONE = "None"
    LISTED_ONLY = "Listed Only"              # Just in Skills section
    MENTIONED = "Mentioned"                  # Mentioned in General / Summary
    PROJECT_EVIDENCE = "Project Evidence"    # Detailed in a Project
    EXPERIENCE_EVIDENCE = "Experience Evidence" # In Work History
    STRONG_IMPLEMENTATION = "Strong Implementation" # Experience/Projects with metrics & details

@dataclass
class RequirementMatchResult:
    requirement_id: str
    requirement_text: str
    priority: str                       # Required / Preferred
    classification: MatchClassification
    matched_term: Optional[str] = None
    canonical_skill: Optional[str] = None
    evidence_sentence: Optional[str] = None
    resume_section: Optional[SectionType] = None
    similarity_score: float = 0.0       # 0.0 to 1.0
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    matching_method: str = "None"       # Exact Canonical, Exact Alias, Fuzzy, Semantic, Related Skill
    explanation: str = ""
    confidence: float = 0.0             # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text,
            "priority": self.priority,
            "classification": self.classification.value,
            "matched_term": self.matched_term,
            "canonical_skill": self.canonical_skill,
            "evidence_sentence": self.evidence_sentence,
            "resume_section": self.resume_section.value if self.resume_section else None,
            "similarity_score": round(self.similarity_score, 3),
            "evidence_strength": self.evidence_strength.value,
            "matching_method": self.matching_method,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 3)
        }
