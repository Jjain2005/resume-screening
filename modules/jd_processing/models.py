from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class RequirementType(str, Enum):
    SKILL = "Skill"
    EXPERIENCE = "Experience"
    EDUCATION = "Education"
    CERTIFICATION = "Certification"
    RESPONSIBILITY = "Responsibility"

class RequirementPriority(str, Enum):
    REQUIRED = "Required"
    PREFERRED = "Preferred"

@dataclass
class RequirementItem:
    id: str
    text: str
    req_type: RequirementType
    priority: RequirementPriority
    canonical_skill: Optional[str] = None
    min_years: Optional[float] = None
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "req_type": self.req_type.value,
            "priority": self.priority.value,
            "canonical_skill": self.canonical_skill,
            "min_years": self.min_years,
            "weight": self.weight
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RequirementItem":
        req_type_str = data.get("req_type", RequirementType.SKILL.value)
        priority_str = data.get("priority", RequirementPriority.REQUIRED.value)
        return cls(
            id=data.get("id", ""),
            text=data.get("text", "").strip(),
            req_type=RequirementType(req_type_str),
            priority=RequirementPriority(priority_str),
            canonical_skill=data.get("canonical_skill"),
            min_years=data.get("min_years"),
            weight=float(data.get("weight", 1.0))
        )

@dataclass
class JobDescription:
    id: str
    title: str
    raw_text: str
    requirements: List[RequirementItem] = field(default_factory=list)

    def get_requirements_by_priority(self, priority: RequirementPriority) -> List[RequirementItem]:
        return [r for r in self.requirements if r.priority == priority]

    def get_requirements_by_type(self, req_type: RequirementType) -> List[RequirementItem]:
        return [r for r in self.requirements if r.req_type == req_type]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "raw_text": self.raw_text,
            "requirements": [r.to_dict() for r in self.requirements]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobDescription":
        return cls(
            id=data.get("id", "jd_default"),
            title=data.get("title", "Job Description"),
            raw_text=data.get("raw_text", ""),
            requirements=[RequirementItem.from_dict(r) for r in data.get("requirements", [])]
        )
