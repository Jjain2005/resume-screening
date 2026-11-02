from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class RequirementType(str, Enum):
    SKILL = "Skill"
    EXPERIENCE = "Experience"
    EDUCATION = "Education"
    CERTIFICATION = "Certification"
    RESPONSIBILITY = "Responsibility"
    BEHAVIOURAL = "Behavioural"  # Generic soft skills / behavioural qualities (communication, teamwork, etc.)

class RequirementPriority(str, Enum):
    MANDATORY = "Mandatory"
    CORE = "Core"
    PREFERRED = "Preferred"
    OPTIONAL = "Optional"
    REQUIRED = "Required"  # Backward-compatible alias for Mandatory

    @property
    def is_mandatory(self) -> bool:
        return self in (RequirementPriority.MANDATORY, RequirementPriority.REQUIRED)

    @property
    def is_core(self) -> bool:
        return self == RequirementPriority.CORE

    @property
    def is_essential(self) -> bool:
        return self in (RequirementPriority.MANDATORY, RequirementPriority.REQUIRED, RequirementPriority.CORE)

    @classmethod
    def from_str(cls, val: str) -> "RequirementPriority":
        if not val:
            return cls.MANDATORY
        v = str(val).strip().lower()
        if v in ("mandatory", "required", "must-have", "must have", "essential", "critical"):
            return cls.MANDATORY
        elif v in ("core", "primary", "key competency", "core competency"):
            return cls.CORE
        elif v in ("preferred", "desired", "plus", "nice to have", "nice-to-have", "good to have"):
            return cls.PREFERRED
        elif v in ("optional", "bonus"):
            return cls.OPTIONAL
        return cls.MANDATORY

class MatchingLogic(str, Enum):
    SINGLE = "Single"
    ALL_OF = "ALL_OF"   # Compound AND: all components required
    ANY_OF = "ANY_OF"   # Compound OR: any component satisfies

@dataclass
class RequirementItem:
    id: str
    text: str
    req_type: RequirementType
    priority: RequirementPriority
    canonical_skill: Optional[str] = None
    min_years: Optional[float] = None
    weight: float = 1.0
    matching_logic: MatchingLogic = MatchingLogic.SINGLE
    compound_components: List[str] = field(default_factory=list)
    accepted_equivalents: List[str] = field(default_factory=list)
    expected_evidence: str = ""
    is_eligibility_condition: bool = False
    importance_rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "req_type": self.req_type.value,
            "priority": self.priority.value,
            "canonical_skill": self.canonical_skill,
            "min_years": self.min_years,
            "weight": round(self.weight, 4),
            "matching_logic": self.matching_logic.value,
            "compound_components": self.compound_components,
            "accepted_equivalents": self.accepted_equivalents,
            "expected_evidence": self.expected_evidence,
            "is_eligibility_condition": self.is_eligibility_condition,
            "importance_rationale": self.importance_rationale
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RequirementItem":
        req_type_str = data.get("req_type", RequirementType.SKILL.value)
        prio_val = data.get("priority", RequirementPriority.MANDATORY.value)
        try:
            priority = RequirementPriority(prio_val)
        except ValueError:
            priority = RequirementPriority.from_str(str(prio_val))

        logic_str = data.get("matching_logic", MatchingLogic.SINGLE.value)
        try:
            matching_logic = MatchingLogic(logic_str)
        except ValueError:
            matching_logic = MatchingLogic.SINGLE

        return cls(
            id=data.get("id", ""),
            text=data.get("text", "").strip(),
            req_type=RequirementType(req_type_str) if req_type_str in RequirementType._value2member_map_ else RequirementType.SKILL,
            priority=priority,
            canonical_skill=data.get("canonical_skill"),
            min_years=data.get("min_years"),
            weight=float(data.get("weight", 1.0)),
            matching_logic=matching_logic,
            compound_components=list(data.get("compound_components", [])),
            accepted_equivalents=list(data.get("accepted_equivalents", [])),
            expected_evidence=str(data.get("expected_evidence", "")),
            is_eligibility_condition=bool(data.get("is_eligibility_condition", False)),
            importance_rationale=str(data.get("importance_rationale", ""))
        )

@dataclass
class JDEvaluationStrategy:
    job_domain: str
    essential_capabilities: List[str] = field(default_factory=list)
    relevant_skills: List[str] = field(default_factory=list)
    expected_practical_knowledge: List[str] = field(default_factory=list)
    desirable_qualifications: List[str] = field(default_factory=list)
    role_summary: str = ""
    strategy_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_domain": self.job_domain,
            "essential_capabilities": list(self.essential_capabilities),
            "relevant_skills": list(self.relevant_skills),
            "expected_practical_knowledge": list(self.expected_practical_knowledge),
            "desirable_qualifications": list(self.desirable_qualifications),
            "role_summary": self.role_summary,
            "strategy_hash": self.strategy_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JDEvaluationStrategy":
        return cls(
            job_domain=data.get("job_domain", "General Technology"),
            essential_capabilities=list(data.get("essential_capabilities", [])),
            relevant_skills=list(data.get("relevant_skills", [])),
            expected_practical_knowledge=list(data.get("expected_practical_knowledge", [])),
            desirable_qualifications=list(data.get("desirable_qualifications", [])),
            role_summary=data.get("role_summary", ""),
            strategy_hash=data.get("strategy_hash", "")
        )

@dataclass
class JobDescription:
    id: str
    title: str
    raw_text: str
    requirements: List[RequirementItem] = field(default_factory=list)
    role_summary: str = ""
    evaluation_strategy: Optional[JDEvaluationStrategy] = None

    def get_requirements_by_priority(self, priority: RequirementPriority) -> List[RequirementItem]:
        if priority in (RequirementPriority.MANDATORY, RequirementPriority.REQUIRED):
            return [r for r in self.requirements if r.priority.is_mandatory]
        return [r for r in self.requirements if r.priority == priority]

    def get_requirements_by_type(self, req_type: RequirementType) -> List[RequirementItem]:
        return [r for r in self.requirements if r.req_type == req_type]

    def normalize_weights(self):
        """
        Dynamically normalizes requirement weights directly based on JD priorities
        so that sum(r.weight) == 1.0. Mandatory requirements are given proportionally
        higher relative weight than Preferred or Optional requirements unless explicit
        weights are already calibrated.
        """
        if not self.requirements:
            return

        total_explicit = sum(r.weight for r in self.requirements)
        # If weights are uninitialized (e.g. all 1.0 or sum == len(reqs)) or zero,
        # assign dynamic relative weights based on priority
        if total_explicit <= 0 or all(r.weight == 1.0 for r in self.requirements):
            for r in self.requirements:
                if r.req_type == RequirementType.RESPONSIBILITY:
                    r.weight = 0.5
                elif r.priority in (RequirementPriority.MANDATORY, RequirementPriority.REQUIRED):
                    r.weight = 3.0
                elif r.priority == RequirementPriority.CORE:
                    r.weight = 2.5
                elif r.priority == RequirementPriority.PREFERRED:
                    r.weight = 1.2
                elif r.priority == RequirementPriority.OPTIONAL:
                    r.weight = 0.5
                else:
                    r.weight = 1.0
            total_explicit = sum(r.weight for r in self.requirements)

        for r in self.requirements:
            r.weight = r.weight / total_explicit

    def validate_matrix(self) -> Dict[str, Any]:
        """
        Validates criteria integrity: ensures weights sum to 1.0, non-empty criteria,
        and reports dynamic criteria dimensions without hardcoding roles.
        """
        if not self.requirements:
            return {"valid": False, "error": "No requirements extracted from Job Description."}

        self.normalize_weights()
        total_w = sum(r.weight for r in self.requirements)
        mandatory_count = sum(1 for r in self.requirements if r.priority.is_mandatory)
        has_min_exp = any(r.min_years is not None and r.min_years > 0 for r in self.requirements)

        return {
            "valid": True,
            "total_requirements": len(self.requirements),
            "mandatory_count": mandatory_count,
            "core_count": sum(1 for r in self.requirements if r.priority == RequirementPriority.CORE),
            "preferred_count": sum(1 for r in self.requirements if r.priority == RequirementPriority.PREFERRED),
            "optional_count": sum(1 for r in self.requirements if r.priority == RequirementPriority.OPTIONAL),
            "has_experience_requirement": has_min_exp,
            "total_weight": round(total_w, 4)
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "raw_text": self.raw_text,
            "role_summary": self.role_summary,
            "evaluation_strategy": self.evaluation_strategy.to_dict() if self.evaluation_strategy else None,
            "requirements": [r.to_dict() for r in self.requirements]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobDescription":
        strat_data = data.get("evaluation_strategy")
        strategy = JDEvaluationStrategy.from_dict(strat_data) if strat_data else None
        jd = cls(
            id=data.get("id", "jd_default"),
            title=data.get("title", "Job Position"),
            raw_text=data.get("raw_text", ""),
            role_summary=data.get("role_summary", ""),
            evaluation_strategy=strategy,
            requirements=[RequirementItem.from_dict(r) for r in data.get("requirements", [])]
        )
        jd.normalize_weights()
        return jd
