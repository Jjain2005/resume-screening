from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class RelationshipType(str, Enum):
    EQUIVALENT = "equivalent"
    RELATED_NOT_EQUIVALENT = "related_not_equivalent"
    RELATED = "related"
    SUBFIELD_OF = "subfield_of"
    PARENT_OF = "parent_of"
    TOOL_OF = "tool_of"
    APPLIED_TO = "applied_to"
    COMPLEMENTARY = "complementary"
    REQUIRES = "requires"
    SUPERSET_OF = "superset_of"

@dataclass
class SkillRelationship:
    target: str
    type: RelationshipType

    @classmethod
    def from_dict(cls, data: dict) -> "SkillRelationship":
        raw_type = data.get("type", "related")
        try:
            rel_type = RelationshipType(raw_type)
        except ValueError:
            rel_type = RelationshipType.RELATED
        return cls(target=data.get("target", "").strip(), type=rel_type)

@dataclass
class SkillEntry:
    id: str
    canonical_name: str
    category: str
    description: str
    aliases: List[str] = field(default_factory=list)
    relationships: List[SkillRelationship] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SkillEntry":
        relationships = [
            SkillRelationship.from_dict(r) for r in data.get("relationships", [])
        ]
        return cls(
            id=data.get("id", ""),
            canonical_name=data.get("canonical_name", "").strip(),
            category=data.get("category", "General"),
            description=data.get("description", ""),
            aliases=[a.strip() for a in data.get("aliases", [])],
            relationships=relationships,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "category": self.category,
            "description": self.description,
            "aliases": self.aliases,
            "relationships": [
                {"target": r.target, "type": r.type.value}
                for r in self.relationships
            ],
        }
