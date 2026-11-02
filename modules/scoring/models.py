from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class Recommendation(str, Enum):
    RECOMMENDED_FOR_SELECTION = "Recommended for Selection"
    FURTHER_REVIEW = "Further Review"            # Borderline — automated follow-up
    NOT_RECOMMENDED = "Not Recommended"
    UNABLE_TO_EVALUATE = "Unable to Evaluate"    # Extraction / API failure — system issue only

    # Backward-compatible aliases
    STRONGLY_RECOMMENDED = "Recommended for Selection"
    RECOMMENDED = "Recommended for Selection"
    MANUAL_REVIEW_REQUIRED = "Further Review"    # Legacy alias — maps to FURTHER_REVIEW
    BORDERLINE = "Further Review"
    LOW_MATCH = "Not Recommended"

@dataclass
class ScoringWeights:
    """
    Universal evaluation weights.
    In the universal JD-driven engine, requirement weights are derived
    dynamically from the Job Description itself.
    """
    essential_skills_weight: float = 0.45
    preferred_skills_weight: float = 0.15
    practical_evidence_weight: float = 0.20
    relevant_experience_weight: float = 0.10
    project_depth_weight: float = 0.10
    llm_evaluation_weight: float = 0.00

    def __init__(
        self,
        essential_skills_weight: Optional[float] = None,
        preferred_skills_weight: Optional[float] = None,
        practical_evidence_weight: Optional[float] = None,
        relevant_experience_weight: Optional[float] = None,
        project_depth_weight: Optional[float] = None,
        llm_evaluation_weight: Optional[float] = None,
        # Backward-compatible parameters
        required_skills_weight: Optional[float] = None,
        evidence_depth_weight: Optional[float] = None,
        experience_project_weight: Optional[float] = None,
        **kwargs
    ):
        if required_skills_weight is not None and essential_skills_weight is None:
            self.essential_skills_weight = required_skills_weight
        else:
            self.essential_skills_weight = essential_skills_weight if essential_skills_weight is not None else 0.45

        self.preferred_skills_weight = preferred_skills_weight if preferred_skills_weight is not None else 0.15

        if evidence_depth_weight is not None and practical_evidence_weight is None:
            self.practical_evidence_weight = evidence_depth_weight
        else:
            self.practical_evidence_weight = practical_evidence_weight if practical_evidence_weight is not None else 0.20

        if experience_project_weight is not None and relevant_experience_weight is None:
            self.relevant_experience_weight = experience_project_weight * 0.5
            self.project_depth_weight = experience_project_weight * 0.5
        else:
            self.relevant_experience_weight = relevant_experience_weight if relevant_experience_weight is not None else 0.10
            self.project_depth_weight = project_depth_weight if project_depth_weight is not None else 0.10

        self.llm_evaluation_weight = llm_evaluation_weight if llm_evaluation_weight is not None else 0.00

    @property
    def required_skills_weight(self) -> float:
        return self.essential_skills_weight

    @property
    def evidence_depth_weight(self) -> float:
        return self.practical_evidence_weight

    @property
    def experience_project_weight(self) -> float:
        return self.relevant_experience_weight + self.project_depth_weight

    def validate(self):
        """Validates weights sum close to 1.0."""
        total = (
            self.essential_skills_weight
            + self.preferred_skills_weight
            + self.practical_evidence_weight
            + self.relevant_experience_weight
            + self.project_depth_weight
            + self.llm_evaluation_weight
        )
        if abs(total - 1.0) > 0.05:
            # Rescale weights proportionally to ensure sum is exactly 1.0
            if total > 0:
                self.essential_skills_weight /= total
                self.preferred_skills_weight /= total
                self.practical_evidence_weight /= total
                self.relevant_experience_weight /= total
                self.project_depth_weight /= total
                self.llm_evaluation_weight /= total

@dataclass
class CandidateEvaluation:
    overall_score: float                              # 0.0 to 100.0
    essential_coverage_pct: float                     # 0.0 to 100.0
    preferred_coverage_pct: float                    # 0.0 to 100.0
    practical_evidence_score: float                  # 0.0 to 100.0
    experience_duration_score: float                 # 0.0 to 100.0
    project_depth_score: float                       # 0.0 to 100.0
    unsupported_keyword_penalty: float               # 0.0 to 10.0 deduction
    verified_tenure_years: float                     # parsed years
    recommendation: Recommendation
    total_essential_count: int
    matched_essential_count: int
    total_preferred_count: int
    matched_preferred_count: int
    score_breakdown_formula: str
    decision_rationale: str
    llm_score: Optional[float] = None
    llm_recommendation: Optional[str] = None
    llm_synthesis: Optional[str] = None
    min_years_required: Optional[float] = None
    transferable_skills: List[Dict[str, str]] = field(default_factory=list)
    is_low_confidence_extraction: bool = False
    is_system_failure: bool = False
    total_mandatory_count: int = 0
    matched_mandatory_count: int = 0
    total_core_count: int = 0
    matched_core_count: int = 0
    demonstrated_essential_count: int = 0
    listed_only_essential_count: int = 0
    dynamic_criteria_scores: Dict[str, float] = field(default_factory=dict)

    # Backward-compatible property accessors
    @property
    def required_coverage_pct(self) -> float:
        return self.essential_coverage_pct

    @property
    def evidence_quality_score(self) -> float:
        return self.practical_evidence_score

    @property
    def experience_relevance_score(self) -> float:
        return self.experience_duration_score

    @property
    def total_required_count(self) -> int:
        return self.total_essential_count

    @property
    def matched_required_count(self) -> int:
        return self.matched_essential_count

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "essential_coverage_pct": round(self.essential_coverage_pct, 1),
            "preferred_coverage_pct": round(self.preferred_coverage_pct, 1),
            "practical_evidence_score": round(self.practical_evidence_score, 1),
            "experience_duration_score": round(self.experience_duration_score, 1),
            "project_depth_score": round(self.project_depth_score, 1),
            "unsupported_keyword_penalty": round(self.unsupported_keyword_penalty, 1),
            "verified_tenure_years": round(self.verified_tenure_years, 1),
            "min_years_required": self.min_years_required,
            "recommendation": self.recommendation.value,
            "total_essential_count": self.total_essential_count,
            "matched_essential_count": self.matched_essential_count,
            "total_preferred_count": self.total_preferred_count,
            "matched_preferred_count": self.matched_preferred_count,
            "transferable_skills": self.transferable_skills,
            "is_low_confidence_extraction": self.is_low_confidence_extraction,
            "is_system_failure": self.is_system_failure,
            "total_mandatory_count": self.total_mandatory_count,
            "matched_mandatory_count": self.matched_mandatory_count,
            "total_core_count": self.total_core_count,
            "matched_core_count": self.matched_core_count,
            "demonstrated_essential_count": self.demonstrated_essential_count,
            "listed_only_essential_count": self.listed_only_essential_count,
            "score_breakdown_formula": self.score_breakdown_formula,
            "decision_rationale": self.decision_rationale,
            "llm_score": round(self.llm_score, 1) if self.llm_score is not None else None,
            "llm_recommendation": self.llm_recommendation,
            "llm_synthesis": self.llm_synthesis,
            "dynamic_criteria_scores": self.dynamic_criteria_scores,
            # Legacy keys for backward compatibility
            "required_coverage_pct": round(self.essential_coverage_pct, 1),
            "evidence_quality_score": round(self.practical_evidence_score, 1),
            "experience_relevance_score": round(self.experience_duration_score, 1),
            "total_required_count": self.total_essential_count,
            "matched_required_count": self.matched_essential_count
        }
