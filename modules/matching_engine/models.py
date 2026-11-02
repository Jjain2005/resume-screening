from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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

class SkillEvidenceStatus(str, Enum):
    DEMONSTRATED = "Demonstrated"
    PARTIAL = "Partial"
    LISTED_ONLY = "Listed Only"
    TRANSFERABLE = "Transferable"
    NOT_FOUND = "Not Found"
    UNCERTAIN = "Uncertain"

@dataclass
class RequirementMatchResult:
    requirement_id: str
    requirement_text: str
    priority: str                       # Mandatory / Core / Preferred / Optional
    classification: MatchClassification
    matched_term: Optional[str] = None
    canonical_skill: Optional[str] = None
    evidence_sentence: Optional[str] = None
    resume_section: Optional[SectionType] = None
    similarity_score: float = 0.0       # 0.0 to 1.0
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    matching_method: str = "None"       # Exact Canonical, Exact Alias, Fuzzy, Semantic, Compound
    explanation: str = ""
    confidence: float = 0.0             # 0.0 to 1.0
    weight: float = 0.0                 # Dynamic normalized weight from JD
    req_type: str = "Skill"
    matching_logic: str = "Single"      # Single, ALL_OF, ANY_OF
    is_eligibility_condition: bool = False
    fulfillment_ratio: float = 0.0      # 0.0 to 1.0
    evidence_passages: List[Dict[str, Any]] = field(default_factory=list)
    component_matches: Dict[str, Any] = field(default_factory=dict)
    evidence_status: Optional[SkillEvidenceStatus] = None

    def __post_init__(self):
        if self.evidence_status is None:
            self.evidence_status = self.derive_evidence_status()

    def derive_evidence_status(self) -> SkillEvidenceStatus:
        if self.classification == MatchClassification.NOT_FOUND:
            return SkillEvidenceStatus.NOT_FOUND
        elif self.classification == MatchClassification.RELATED_SKILL_NOT_EQUIVALENT:
            return SkillEvidenceStatus.TRANSFERABLE
        elif self.classification == MatchClassification.NEEDS_MANUAL_REVIEW:
            return SkillEvidenceStatus.UNCERTAIN
        elif self.classification == MatchClassification.PARTIAL_EVIDENCE:
            return SkillEvidenceStatus.PARTIAL
        elif self.classification in (MatchClassification.EXACT_MATCH, MatchClassification.STRONG_SEMANTIC_MATCH, MatchClassification.FUZZY_MATCH):
            if self.evidence_strength in (EvidenceStrength.STRONG_IMPLEMENTATION, EvidenceStrength.EXPERIENCE_EVIDENCE, EvidenceStrength.PROJECT_EVIDENCE):
                return SkillEvidenceStatus.DEMONSTRATED
            elif self.evidence_strength in (EvidenceStrength.LISTED_ONLY, EvidenceStrength.MENTIONED):
                return SkillEvidenceStatus.LISTED_ONLY
            return SkillEvidenceStatus.DEMONSTRATED
        return SkillEvidenceStatus.NOT_FOUND

    def to_dict(self) -> dict:
        status_val = self.evidence_status.value if self.evidence_status else self.derive_evidence_status().value
        return {
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text,
            "priority": self.priority,
            "classification": self.classification.value,
            "evidence_status": status_val,
            "matched_term": self.matched_term,
            "canonical_skill": self.canonical_skill,
            "evidence_sentence": self.evidence_sentence,
            "resume_section": self.resume_section.value if self.resume_section else None,
            "similarity_score": round(self.similarity_score, 3),
            "evidence_strength": self.evidence_strength.value,
            "matching_method": self.matching_method,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 3),
            "weight": round(self.weight, 4),
            "req_type": self.req_type,
            "matching_logic": self.matching_logic,
            "is_eligibility_condition": self.is_eligibility_condition,
            "fulfillment_ratio": round(self.fulfillment_ratio, 3),
            "evidence_passages": self.evidence_passages,
            "component_matches": self.component_matches
        }
